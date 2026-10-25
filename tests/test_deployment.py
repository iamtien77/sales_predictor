"""
test_deployment.py - Kiểm thử module triển khai dự đoán theo kịch bản
======================================================================
- test_predict_sales(): Kiểm tra hàm nhận vector features và trả về doanh số hợp lệ (số thực >= 0)
- test_predict_new_product_scenario(): Kiểm tra kịch bản sản phẩm mới ra mắt
- test_predict_marketing_campaign_scenario(): Kiểm tra kịch bản thay đổi ngân sách quảng cáo
"""
"""Kiểm thử với mô hình thật fit trên fixture nhỏ, không phải mô hình bàn giao."""
import pickle
import numpy as np
import pandas as pd
import pytest
from src import config
from src.data.feature_engineering import WeeklySalesFeatureEngineer
from src.data.preprocessing import CustomDataPipeline
from src.models.tree import DecisionTreeRegressorFromScratch
from src.models.persistence import save_artifacts
from src.deployment.predictor import SalesPredictor, to_brl, demo_prediction
from src.data.launch import HISTORY_COLUMNS
from src.deployment.predictor import validate_inputs


@pytest.fixture
def service():
    rows = []
    for i in range(8):
        rows.append({'product_id': f'p{i}', 'category': 'books' if i % 2 else 'toys',
                     'price': 10.0 + i, config.TIMESTAMP_COLUMN: '2026-10-26',
                     **{c: 0.0 for c in HISTORY_COLUMNS}, 'sales': float(i * 10)})
    frame = pd.DataFrame(rows)
    pipeline = CustomDataPipeline(num_cols=config.WEEKLY_NUMERICAL_COLS,
                                  cat_cols=['category'], outlier_cols=[],
                                  feature_engineer=WeeklySalesFeatureEngineer())
    X, y = pipeline.fit_transform(frame)
    model = DecisionTreeRegressorFromScratch(max_depth=3, min_samples_leaf=1)
    model.fit(X, y)
    return SalesPredictor(model, pipeline), frame.drop(columns='sales')


def test_prediction_roundtrip_and_batch(service, tmp_path):
    predictor, frame = service
    before = pickle.dumps(predictor.pipeline)
    copy = frame.copy(deep=True)
    batch = predictor.predict(frame)
    singles = [predictor.predict(row.to_dict()).predicted_revenue_brl.iloc[0] for _, row in frame.iterrows()]
    np.testing.assert_allclose(batch.predicted_revenue_brl, singles)
    pd.testing.assert_frame_equal(frame, copy)
    assert pickle.dumps(predictor.pipeline) == before
    path = save_artifacts(predictor.pipeline, predictor.model, tmp_path / 'model.pkl')
    loaded = SalesPredictor.from_artifact(path)
    np.testing.assert_allclose(loaded.predict(frame).predicted_revenue_brl, singles)
    X, _ = predictor.pipeline.transform(frame)
    np.testing.assert_allclose(batch.predicted_revenue_brl, np.maximum(0, np.expm1(predictor.model.predict(X))))


@pytest.mark.parametrize('column,value', [('price', 0), ('price', -1), ('price', True),
    ('price', np.nan), ('price', np.inf), ('last_week_sales', -1),
    ('past_review_rating', 6), ('price', 'abc'), ('category', ''),
    (config.TIMESTAMP_COLUMN, '2026-10-27')])
def test_reject_invalid_input(service, column, value):
    predictor, frame = service
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        predictor.predict(frame)


def test_missing_history_empty_and_unknown_category(service):
    predictor, frame = service
    with pytest.raises(ValueError, match='Thiếu cột'):
        predictor.predict(frame.drop(columns='last_week_sales'))
    with pytest.raises(ValueError):
        predictor.predict(frame.iloc[:0])
    frame['category'] = 'never_seen'
    assert np.isfinite(predictor.predict(frame).predicted_revenue_brl).all()
    encoded = predictor.pipeline.encoder.transform(frame)
    assert (encoded == 0).all()


def test_target_does_not_influence_prediction(service):
    predictor, frame = service
    expected = predictor.predict(frame).predicted_revenue_brl
    frame['sales'] = 1e15
    frame['log_sales'] = 1e10
    np.testing.assert_allclose(predictor.predict(frame).predicted_revenue_brl, expected)


def test_reverse_transform():
    np.testing.assert_allclose(to_brl([np.log1p(100)]), [100])
    np.testing.assert_allclose(to_brl([100], False), [100])
    assert to_brl([-1])[0] == 0
    for value in [np.nan, np.inf, 1000]:
        with pytest.raises(ValueError): to_brl([value])


def test_artifact_errors(service, tmp_path):
    with pytest.raises(FileNotFoundError): SalesPredictor.from_artifact(tmp_path / 'missing.pkl')
    path = tmp_path / 'invalid.pkl'
    path.write_bytes(pickle.dumps({'model': 1}))
    with pytest.raises(ValueError): SalesPredictor.from_artifact(path)


"""Kiểm thử UI, giới hạn CSV, xuất bảng an toàn và hiển thị lỗi có kiểm soát."""
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
import app as ui
from app import read_batch_csv, csv_bytes, load_service
from src import config
from src.models.persistence import save_artifacts

ROOT = Path(__file__).resolve().parents[1]


def test_csv_roundtrip():
    frame = pd.DataFrame({'category': ['đồ chơi'], 'price': [10]})
    pd.testing.assert_frame_equal(read_batch_csv(csv_bytes(frame)), frame)
    for content in [b'', b'\xff\xff', b'a,a\n1,2', b',b\n1,2']:
        with pytest.raises(ValueError): read_batch_csv(content)


def test_csv_size_row_width_and_cell_limits(monkeypatch):
    with pytest.raises(ValueError, match='10 MB'): read_batch_csv(b'x' * (ui.MAX_UPLOAD_BYTES + 1))
    monkeypatch.setattr(ui, 'MAX_BATCH_ROWS', 2)
    with pytest.raises(ValueError, match='dòng'): read_batch_csv(b'a\n1\n2\n3')
    with pytest.raises(ValueError, match='100 cột'):
        read_batch_csv((','.join(f'c{i}' for i in range(101)) + '\n' + ','.join(['1'] * 101)).encode())
    with pytest.raises(ValueError, match='4.096'): read_batch_csv(('a\n' + 'x' * 4097).encode())


@pytest.mark.parametrize('value', ['=1+1', '+SUM(A1)', '-1+2', '@SUM(A1)', '  =1', '\t=1', '\r=1', '\n=1'])
def test_csv_formula_injection(value):
    frame = pd.DataFrame({'text': [value], 'number': [-10]})
    original = frame.copy(deep=True)
    exported = pd.read_csv(__import__('io').BytesIO(csv_bytes(frame)))
    assert exported['text'].iloc[0] == "'" + value
    assert exported['number'].iloc[0] == -10
    pd.testing.assert_frame_equal(frame, original)


def test_header_formula_sanitized():
    assert b"'=SUM(A1)" in csv_bytes(pd.DataFrame({'=SUM(A1)': ['abc']}))


def test_artifact_path_restricted():
    with pytest.raises(ValueError): load_service('../model.pkl')


def test_app_no_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SAVED_MODELS_DIR', tmp_path)
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=20)
    assert not app.exception
    assert app.button[0].disabled
    visible = ' '.join(x.value for x in app.markdown)
    assert 'results/saved_models' not in visible
    assert 'Task 2' not in visible
    assert not app.text_input[0].value
    app.radio[0].set_value('Dự báo hàng loạt').run()
    assert not app.exception
    assert next(b for b in app.button if b.label == 'Dự báo cả lô').disabled
    app.radio[0].set_value('Mô phỏng kinh doanh').run()
    app.selectbox[0].select('Ngân sách marketing').run()
    next(b for b in app.button if b.label == 'Lập bảng ngân sách').click().run()
    assert not app.exception
    assert any('Doanh thu tăng thêm (BRL)' in t.value for t in app.dataframe)
    app.selectbox[0].select('Giảm giá Black Friday').run()
    next(b for b in app.button if b.label == 'Tính doanh thu').click().run()
    assert not app.exception
    assert any('Doanh thu giả định (BRL)' in t.value for t in app.dataframe)


def test_app_with_fixture_artifact(service, tmp_path, monkeypatch):
    predictor, _ = service
    save_artifacts(predictor.pipeline, predictor.model, tmp_path / 'fixture.pkl')
    monkeypatch.setattr(config, 'SAVED_MODELS_DIR', tmp_path)
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=20)
    next(t for t in app.text_input if t.label == 'Mã sản phẩm').set_value('SP001')
    next(t for t in app.text_input if t.label == 'Danh mục').set_value('toys')
    next(b for b in app.button if b.label == 'Dự báo doanh thu').click().run()
    assert not app.exception
    assert any(m.label == 'Doanh thu dự báo (BRL)' for m in app.metric)
    app.radio[0].set_value('Mô phỏng kinh doanh').run()
    next(t for t in app.text_input if t.label == 'Danh mục sản phẩm mới').set_value('toys')
    next(b for b in app.button if b.label == 'Ước lượng doanh thu').click().run()
    assert not app.exception
    assert any(m.label == 'Doanh thu tham khảo (BRL)' for m in app.metric)
    (tmp_path / 'fixture.pkl').unlink()
    app.radio[0].set_value('Dự báo sản phẩm').run()
    assert not app.metric  # Không để kết quả cũ khi mô hình bị gỡ.


def test_system_error_does_not_expose_details(service, tmp_path, monkeypatch):
    predictor, _ = service
    save_artifacts(predictor.pipeline, predictor.model, tmp_path / 'fixture.pkl')
    monkeypatch.setattr(config, 'SAVED_MODELS_DIR', tmp_path)
    class BrokenService:
        def predict(self, features):
            raise RuntimeError('/private/path SECRET_TOKEN')
    monkeypatch.setattr(ui.SalesPredictor, 'from_artifact', lambda p: BrokenService())
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=20)
    next(b for b in app.button if b.label == 'Dự báo doanh thu').click().run()
    assert not app.exception
    assert app.error
    assert all('SECRET_TOKEN' not in e.value and '/private/path' not in e.value for e in app.error)


def test_local_security_configuration():
    import tomllib
    settings = tomllib.loads((ROOT / '.streamlit/config.toml').read_text())
    assert settings['server']['address'] == '127.0.0.1'
    assert settings['server']['enableCORS'] is True
    assert settings['server']['enableXsrfProtection'] is True
    assert settings['server']['maxUploadSize'] == 10
    assert settings['browser']['gatherUsageStats'] is False


def test_demo_zero_actual_has_no_infinite_percentage(service):
    predictor, frame = service
    X, _ = predictor.pipeline.transform(frame.iloc[:1])
    result = demo_prediction(predictor.model, predictor.pipeline, X, np.zeros(1), 1)
    assert result.actual_revenue_brl.iloc[0] == 0
    assert pd.isna(result['Error_Percentage_%'].iloc[0])


def test_discount_json_download_uses_null_for_undefined_values(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(config, 'SAVED_MODELS_DIR', tmp_path)
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=20)
    app.radio[0].set_value('Mô phỏng kinh doanh').run()
    app.selectbox[0].select('Giảm giá Black Friday').run()
    next(b for b in app.button if b.label == 'Tính doanh thu').click().run()
    assert not app.exception
    assert len(app.get('download_button')) == 2
    from src.deployment.predictor import discount_scenario
    records = json.loads(discount_scenario(100, [0, .2, 1], 10).to_json(orient='records'))
    assert records[0]['baseline_revenue_brl'] == 1000
    assert records[1]['revenue_change_brl'] == -200
    assert records[2]['units_to_match_reference'] is None
