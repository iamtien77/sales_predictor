#```powershell, 
# python -m venv .venv,
# .\.venv\Scripts\python.exe -m pip install numpy pandas scipy streamlit pytest,
# .\.venv\Scripts\python.exe -m streamlit run app.py

"""Chạy từ thư mục dự án: python -m streamlit run app.py."""
import csv
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.data.launch import HISTORY_COLUMNS
from src.deployment.predictor import InputValidationError
from src.deployment.predictor import SalesPredictor
from src.evaluation.listing_launch import new_product_scenario
from src.deployment.predictor import marketing_scenario, discount_scenario

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_ROWS = 10_000
MAX_BATCH_COLUMNS = 100
MAX_CELL_LENGTH = 4096

LABELS = {
    'product_id': 'Mã sản phẩm', 'category': 'Danh mục',
    'product_category_name': 'Danh mục', 'price': 'Giá bán (BRL)',
    'order_purchase_timestamp': 'Tuần dự báo',
    'past_review_count': 'Số lượt đánh giá trước tuần dự báo',
    'past_review_rating': 'Điểm đánh giá trung bình (0–5)',
    'customer_past_spend': 'Chi tiêu tích lũy của khách hàng (BRL)',
    'customer_past_orders': 'Số đơn hàng tích lũy của khách hàng',
    'customer_state_count': 'Số bang có khách mua hàng',
    'last_week_sales': 'Doanh thu tuần trước (BRL)',
    'sales_lag_2_weeks': 'Doanh thu cách 2 tuần (BRL)',
    'sales_lag_4_weeks': 'Doanh thu cách 4 tuần (BRL)',
    'sales_mean_4_weeks': 'Doanh thu trung bình 4 tuần (BRL)',
    'sales_std_4_weeks': 'Độ lệch chuẩn doanh thu 4 tuần (BRL)',
    'last_week_units': 'Sản lượng tuần trước',
    'weeks_since_first_sale': 'Số tuần từ lần bán đầu tiên',
    'predicted_revenue_brl': 'Doanh thu dự báo (BRL)',
    'ad_spend_brl': 'Ngân sách (BRL)',
    'incremental_revenue_brl': 'Doanh thu tăng thêm (BRL)', 'roi': 'ROI',
    'status': 'Trạng thái', 'discount_rate': 'Tỷ lệ giảm',
    'discounted_price_brl': 'Giá sau giảm (BRL)', 'assumed_units': 'Sản lượng giả định',
    'scenario_revenue_brl': 'Doanh thu giả định (BRL)',
    'reference_revenue_brl': 'Doanh thu tham chiếu (BRL)',
    'baseline_revenue_brl': 'Doanh thu trước giảm giá (BRL)',
    'revenue_change_brl': 'Chênh lệch doanh thu (BRL)',
    'units_to_match_reference': 'Sản lượng cần để đạt tham chiếu',
    'product_weight_g': 'Trọng lượng (g)', 'product_photos_qty': 'Số ảnh sản phẩm',
    'product_name_lenght': 'Độ dài tên sản phẩm',
    'product_description_lenght': 'Độ dài mô tả sản phẩm',
}


class CsvInputError(ValueError):
    """Thông báo có kiểm soát, không chứa đường dẫn hay nội dung file."""


def read_batch_csv(content):
    """Giới hạn dung lượng trước giải mã, số dòng ngay khi đọc và độ rộng bảng."""
    if not isinstance(content, bytes) or not content or len(content) > MAX_UPLOAD_BYTES:
        raise CsvInputError('Chọn CSV không rỗng, dung lượng tối đa 10 MB.')
    try:
        text = content.decode('utf-8-sig')
        header = next(csv.reader(StringIO(text)))
        if not header or any(not c.strip() for c in header) or len(set(header)) != len(header):
            raise CsvInputError('Tên cột không được trống hoặc trùng nhau.')
        if len(header) > MAX_BATCH_COLUMNS:
            raise CsvInputError('CSV chỉ được có tối đa 100 cột.')
        frame = pd.read_csv(StringIO(text), nrows=MAX_BATCH_ROWS + 1)
    except CsvInputError:
        raise
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, StopIteration, ValueError):
        raise CsvInputError('Không đọc được CSV. Kiểm tra mã hóa UTF-8 và định dạng bảng.') from None
    if frame.empty or len(frame) > MAX_BATCH_ROWS:
        raise CsvInputError('CSV cần có từ 1 đến 10.000 dòng dữ liệu.')
    for col in frame.select_dtypes(include=['object', 'string']).columns:
        if frame[col].map(lambda v: isinstance(v, str) and len(v) > MAX_CELL_LENGTH).any():
            raise CsvInputError('Nội dung mỗi ô không được vượt quá 4.096 ký tự.')
    return frame


def spreadsheet_safe(value):
    """Giảm nguy cơ CSV formula injection khi người dùng mở file bằng Excel.

    Chỉ thay đổi chuỗi xuất ra, giữ nguyên số và dữ liệu dùng để dự báo.
    Dấu nháy đơn được thêm vào các chuỗi có thể được hiểu là công thức.
    """
    if isinstance(value, str) and (value.lstrip().startswith(('=', '+', '-', '@'))
                                   or value.startswith(('\t', '\r', '\n'))):
        return "'" + value
    return value


def csv_bytes(frame):
    safe = frame.copy(deep=True)
    safe.columns = [spreadsheet_safe(str(c)) for c in safe.columns]
    for col in safe.columns:
        safe[col] = safe[col].map(spreadsheet_safe)
    return safe.to_csv(index=False).encode('utf-8-sig')


def load_service(artifact_name):
    """Chỉ nạp mô hình từ thư mục nội bộ đã biết, không nhận pickle qua web."""
    folder = config.SAVED_MODELS_DIR.resolve()
    path = (folder / artifact_name).resolve()
    if path.parent != folder or path.suffix != '.pkl':
        raise ValueError('Mô hình không thuộc thư mục được phép.')
    return SalesPredictor.from_artifact(path)


def display_error(error):
    """Chỉ hiện lỗi nhập liệu do lớp kiểm tra của My tạo; giấu lỗi hệ thống."""
    if isinstance(error, CsvInputError):
        st.error(str(error))
    elif isinstance(error, InputValidationError):
        message = str(error)
        for column in sorted(LABELS, key=len, reverse=True):
            message = message.replace(column, LABELS[column])
        st.error(message)
    else:
        st.error('Chưa xử lý được yêu cầu. Kiểm tra dữ liệu đầu vào hoặc liên hệ người quản lý mô hình.')


def show_result(frame, key):
    st.dataframe(frame.rename(columns=LABELS), hide_index=True, use_container_width=True)
    st.download_button('Tải kết quả CSV', csv_bytes(frame), file_name=f'{key}.csv',
                       mime='text/csv', key=f'download_{key}')
    st.download_button('Tải kết quả JSON', frame.to_json(orient='records', force_ascii=False, date_format='iso', indent=2).encode('utf-8'),
                       file_name=f'{key}.json', mime='application/json', key=f'json_{key}')


def apply_design():
    # HTML/CSS cố định, tuyệt đối không ghép dữ liệu người dùng vào markup.
    st.markdown('''<style>
    .stApp {background: #fff; color: #18181b;}
    [data-testid="stSidebar"] {background: #f6f6f6; border-right: 1px solid #e5e5e5;}
    .block-container {max-width: 1220px; padding-top: 2.6rem; padding-bottom: 3rem;}
    h1 {font-size: 2.4rem !important; font-weight: 650 !important; letter-spacing: -.055rem;}
    h2 {font-size: 1.4rem !important; font-weight: 600 !important;}
    h3 {font-size: 1.05rem !important; font-weight: 600 !important;}
    [data-testid="stMetricValue"] {font-size: 2.8rem; letter-spacing: -.07rem;}
    [data-testid="stForm"] {border-color: #e4e4e7; border-radius: 10px; padding: 1.4rem;}
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {font-size: .95rem;}
    [data-testid="stSidebar"] .stRadio label {padding: .45rem 0;}
    .brand {font-size:1.2rem;font-weight:750;letter-spacing:-.04rem;margin:0 0 .2rem;}
    .eyebrow {font-size:.72rem;letter-spacing:.12rem;color:#71717a;margin:0 0 .6rem;}
    .status {display:inline-block;border:1px solid #d4d4d8;border-radius:20px;
             padding:5px 11px;font-size:.75rem;color:#52525b;background:#fff;}
    .empty {padding:2rem 1.2rem;border:1px dashed #d4d4d8;border-radius:10px;
            text-align:center;color:#71717a;background:#fafafa;}
    .empty strong {display:block;color:#27272a;font-size:1.1rem;margin-bottom:.4rem;}
    [data-testid="stAlert"] {background:#f6f6f6;border:1px solid #e5e5e5;color:#27272a;}
    [data-testid="stAlert"] svg {color:#52525b;}
    hr {border-color:#e5e5e5;}
    @media(max-width: 720px) {.block-container {padding-top:1.5rem;} h1 {font-size:1.9rem !important;}}
    </style>''', unsafe_allow_html=True)


def current_service():
    files = sorted(config.SAVED_MODELS_DIR.glob('*.pkl'))
    selected = next((p for p in files if p.name == 'best_sales_model.pkl'), files[0] if files else None)
    service, signature = None, None
    if selected is not None:
        try:
            signature = (selected.name, selected.stat().st_mtime_ns)
            service = load_service(selected.name)
        except Exception:
            signature = None
    if st.session_state.get('artifact_signature') != signature:
        for key in ('single_result', 'batch_result', 'new_result'):
            st.session_state.pop(key, None)
        st.session_state['artifact_signature'] = signature
    return service


def history_fields():
    history = {}
    with st.expander('Lịch sử bán hàng', expanded=True):
        cols = st.columns(2)
        sales_cols = ['last_week_sales', 'sales_lag_2_weeks', 'sales_lag_4_weeks',
                      'sales_mean_4_weeks', 'sales_std_4_weeks', 'last_week_units', 'weeks_since_first_sale']
        for i, col in enumerate(sales_cols):
            with cols[i % 2]:
                history[col] = st.number_input(LABELS[col], min_value=0.0, value=0.0, key='history_' + col)
    with st.expander('Đánh giá và khách hàng'):
        cols = st.columns(2)
        for i, col in enumerate(c for c in HISTORY_COLUMNS if c not in history):
            with cols[i % 2]:
                history[col] = st.number_input(LABELS.get(col, col), min_value=0.0,
                    max_value=5.0 if col == 'past_review_rating' else None, value=0.0, key='history_' + col)
    return history


def render_single(service, monday):
    st.title('Dự báo sản phẩm')
    st.caption('Ước tính doanh thu cho tuần tiếp theo.')
    st.divider()
    left, right = st.columns([1.65, 1], gap='large')
    with left:
        with st.form('single_product'):
            st.subheader('Thông tin sản phẩm')
            a, b = st.columns(2)
            with a:
                product_id = st.text_input('Mã sản phẩm', placeholder='VD: SP001')
                price = st.number_input('Giá bán (BRL)', min_value=0.01, value=100.0)
            with b:
                category = st.text_input('Danh mục', placeholder='VD: informatica_acessorios')
                week = st.date_input('Tuần dự báo · thứ Hai', monday)
            history = history_fields()
            submitted = st.form_submit_button('Dự báo doanh thu', type='primary', disabled=service is None,
                                              use_container_width=True)
        with st.expander('Hướng dẫn nhập liệu'):
            st.write('Dùng số liệu đã có trước tuần dự báo. Chỉ nhập 0 khi lịch sử thực tế bằng 0. Sản phẩm chưa có lịch sử: chọn Mô phỏng kinh doanh → Sản phẩm mới.')
    if submitted:
        try:
            st.session_state['single_result'] = service.predict({'product_id': product_id,
                'category': category, 'price': price, config.TIMESTAMP_COLUMN: str(week), **history})
        except Exception as error:
            st.session_state.pop('single_result', None)
            with left: display_error(error)
    with right:
        st.subheader('Kết quả dự báo')
        if service is None:
            st.markdown('<div class="empty"><strong>Chưa sẵn sàng dự báo</strong>Vui lòng thử lại khi mô hình được cập nhật.</div>', unsafe_allow_html=True)
        elif 'single_result' not in st.session_state:
            st.markdown('<div class="empty"><strong>Chưa có kết quả</strong>Nhập thông tin và chọn Dự báo doanh thu.</div>', unsafe_allow_html=True)
        else:
            result = st.session_state['single_result']
            st.metric('Doanh thu dự báo (BRL)', f"{result['predicted_revenue_brl'].iloc[0]:,.2f}")
            st.caption('Tuần bắt đầu ' + pd.Timestamp(result[config.TIMESTAMP_COLUMN].iloc[0]).strftime('%d/%m/%Y'))
            show_result(result[['product_id', 'category', config.TIMESTAMP_COLUMN, 'predicted_revenue_brl']], 'single')


def render_batch(service, monday):
    st.title('Dự báo hàng loạt')
    st.caption('Tải danh sách sản phẩm và nhận kết quả trong một lần.')
    st.divider()
    left, right = st.columns([1.3, 1], gap='large')
    with left:
        st.subheader('Danh sách sản phẩm')
        uploaded = st.file_uploader('Chọn tệp CSV', type=['csv'], help='UTF-8 · tối đa 10 MB và 10.000 dòng')
        if st.button('Dự báo cả lô', type='primary', disabled=service is None or uploaded is None):
            try:
                # Kiểm tra size trước getvalue để tránh sao chép file lớn.
                if uploaded.size > MAX_UPLOAD_BYTES:
                    raise CsvInputError('Dung lượng CSV tối đa 10 MB.')
                frame = read_batch_csv(uploaded.getvalue())
                st.session_state['batch_result'] = service.predict(frame)
            except Exception as error:
                st.session_state.pop('batch_result', None)
                display_error(error)
    with right:
        st.subheader('Chuẩn bị dữ liệu')
        st.write('Dùng tệp mẫu để giữ đúng tên cột và định dạng tuần.')
        template = pd.DataFrame([{'product_id': 'SP001', 'category': 'informatica_acessorios',
            'price': 100.0, config.TIMESTAMP_COLUMN: str(monday), **{c: 0.0 for c in HISTORY_COLUMNS}}])
        st.download_button('Tải CSV mẫu', csv_bytes(template), file_name='prediction_input_template.csv', mime='text/csv')
        st.caption('Thay dòng minh họa bằng số liệu thực tế trước khi dự báo.')
    if service is not None and 'batch_result' in st.session_state:
        st.divider()
        st.subheader('Kết quả')
        result = st.session_state['batch_result']
        a, b = st.columns(2)
        a.metric('Số sản phẩm / dòng', len(result))
        b.metric('Tổng doanh thu dự báo (BRL)', f"{result['predicted_revenue_brl'].sum():,.2f}")
        show_result(result, 'batch')


def render_scenarios(service, monday):
    st.title('Mô phỏng kinh doanh')
    st.caption('Khám phá các phương án trước khi đưa ra quyết định.')
    st.divider()
    scenario = st.selectbox('Kịch bản', ['Sản phẩm mới', 'Ngân sách marketing', 'Giảm giá Black Friday'])
    if scenario == 'Sản phẩm mới':
        left, right = st.columns([1.5, 1], gap='large')
        with left:
            with st.form('new_product'):
                st.subheader('Sản phẩm mới')
                a, b = st.columns(2)
                with a:
                    category = st.text_input('Danh mục sản phẩm mới', placeholder='VD: informatica_acessorios')
                    price = st.number_input('Giá sản phẩm mới (BRL)', min_value=0.01, value=100.0)
                with b:
                    week = st.date_input('Tuần dự báo sản phẩm mới', monday)
                spec = {}
                if service is not None and service.cold_start_model is not None:
                    with st.expander('Thông tin bổ sung'):
                        for col in ['product_weight_g', 'product_photos_qty', 'product_name_lenght', 'product_description_lenght']:
                            spec[col] = st.number_input(LABELS[col], min_value=0.0, value=0.0)
                run = st.form_submit_button('Ước lượng doanh thu', type='primary', disabled=service is None)
            if run:
                try:
                    st.session_state['new_result'] = new_product_scenario({'category': category, 'price': price,
                        config.TIMESTAMP_COLUMN: str(week), **spec}, service)
                except Exception as error:
                    st.session_state.pop('new_result', None)
                    display_error(error)
        with right:
            st.subheader('Doanh thu tham khảo')
            if service is not None and 'new_result' in st.session_state:
                result = st.session_state['new_result']
                st.metric('Doanh thu tham khảo (BRL)', f"{result['predicted_revenue_brl']:,.2f}")
                st.caption('Ước lượng cho sản phẩm mới; chưa có lịch sử bán hàng.')
                if result['method'] == 'weekly_model_zero_history':
                    st.caption('Kết quả tham khảo với lịch sử bằng 0; chưa kiểm định riêng cho tuần ra mắt.')
                st.download_button('Tải kết quả CSV', csv_bytes(pd.DataFrame([result])), file_name='new_product.csv', mime='text/csv')
                st.download_button('Tải kết quả JSON', pd.DataFrame([result]).to_json(orient='records', force_ascii=False, indent=2).encode('utf-8'), file_name='new_product.json', mime='application/json')
            else:
                st.caption('Kết quả sẽ xuất hiện sau khi có mô hình và thông tin sản phẩm.')
    elif scenario == 'Ngân sách marketing':
        st.subheader('Kế hoạch ngân sách')
        st.caption('Chưa đủ dữ liệu để ước lượng doanh thu tăng thêm hoặc ROI.')
        with st.form('marketing'):
            levels = st.text_input('Các mức ngân sách (BRL)', '100,500,1000', help='Phân tách bằng dấu phẩy.')
            run = st.form_submit_button('Lập bảng ngân sách', type='primary')
        if run:
            try:
                if len(levels) > 4096 or levels.count(',') >= 100:
                    raise CsvInputError('Nhập tối đa 100 mức ngân sách.')
                show_result(marketing_scenario(levels.split(',')), 'marketing')
            except Exception as error: display_error(error)
    else:
        st.subheader('Giảm giá Black Friday')
        st.caption('Doanh thu tính theo sản lượng giả định, không phải dự báo tác động giảm giá.')
        with st.form('discount'):
            a, b = st.columns(2)
            with a:
                price = st.number_input('Giá gốc (BRL)', min_value=0.01, value=100.0)
                units = st.number_input('Sản lượng giả định', min_value=0, value=10, step=1)
            with b:
                reference = st.number_input('Doanh thu tham chiếu (BRL)', min_value=0.0, value=1000.0)
                rates = st.multiselect('Mức giảm giá (%)', [0, 10, 20, 30, 40, 50, 100], default=[0, 10, 20, 30, 50])
            run = st.form_submit_button('Tính doanh thu', type='primary')
        if run:
            try:
                result = discount_scenario(price, [r / 100 for r in rates], units, reference)
                show_result(result.drop(columns='method'), 'discount')
                if 100 in rates: st.caption('Giảm 100% không thể đạt doanh thu tham chiếu dương; ô sản lượng tương ứng để trống.')
            except Exception as error: display_error(error)


def render_app():
    st.set_page_config(page_title='Sales Predictor', page_icon='◼', layout='wide')
    apply_design()
    service = current_service()
    with st.sidebar:
        st.markdown('<p class="brand">Sales Predictor</p><p class="eyebrow">DỰ BÁO DOANH THU</p>', unsafe_allow_html=True)
        st.divider()
        page = st.radio('Chức năng', ['Dự báo sản phẩm', 'Dự báo hàng loạt', 'Mô phỏng kinh doanh'], label_visibility='collapsed')
        st.divider()
        status = 'Sẵn sàng dự báo' if service is not None else 'Chưa sẵn sàng dự báo'
        st.markdown('<span class="status">' + status + '</span>', unsafe_allow_html=True)
        st.caption('Theo tuần · Đơn vị BRL')
    today = date.today()
    monday = today + timedelta(days=(7 - today.weekday()) % 7)
    if page == 'Dự báo sản phẩm': render_single(service, monday)
    elif page == 'Dự báo hàng loạt': render_batch(service, monday)
    else: render_scenarios(service, monday)


if __name__ == '__main__':
    render_app()
