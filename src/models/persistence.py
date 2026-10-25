"""
persistence.py - Lưu và tải Model / Pipeline Artifacts thuần túy (Pickle)
===========================================================================
Đảm bảo tính tương thích và khép kín khi triển khai (Deployment):
  - save_artifacts(pipeline, model, filepath)
  - load_artifacts(filepath)
  - save_model(model, scaler, filepath)
  - load_model(filepath)
  - save_pipeline(pipeline, filepath)
  - load_pipeline(filepath)
"""
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config


def save_artifacts(
    pipeline: Any,
    model: Any,
    filepath: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    cold_start_model: Optional[Any] = None
) -> Path:
    """
    Serialize cả bộ tiền xử lý (CustomDataPipeline), Mô hình hồi quy chính đã fit,
    và Mô hình Cold-Start (nếu có) vào một gói artifact (.pkl) duy nhất để phục vụ deployment.
    """
    if filepath is None:
        filepath = config.SAVED_MODELS_DIR / "best_sales_model.pkl"
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline": pipeline,
        "model": model,
        "model_type": model.__class__.__name__,
        "cold_start_model": cold_start_model,
        "saved_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }

    with open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return filepath


def load_artifacts(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """Tải gói artifact đã lưu từ đĩa"""
    if filepath is None:
        filepath = config.SAVED_MODELS_DIR / "best_sales_model.pkl"
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file artifact tại: {filepath}")

    with open(filepath, "rb") as f:
        payload = pickle.load(f)

    return payload


def save_model(model: Any, scaler: Optional[Any] = None, filepath: Optional[Path] = None) -> Path:
    """Lưu riêng model và scaler (tương thích ngược API)"""
    if filepath is None:
        filepath = config.SAVED_MODELS_DIR / "model_bundle.pkl"
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def load_model(filepath: Optional[Path] = None) -> Tuple[Any, Optional[Any]]:
    """Tải model và scaler (tương thích ngược API)"""
    if filepath is None:
        filepath = config.SAVED_MODELS_DIR / "model_bundle.pkl"
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "model" in data:
        return data["model"], data.get("scaler")
    return data, None


def save_pipeline(pipeline: Any, filepath: Optional[Path] = None) -> Path:
    """Lưu riêng pipeline tiền xử lý"""
    if filepath is None:
        filepath = config.SAVED_MODELS_DIR / "pipeline.pkl"
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def load_pipeline(filepath: Optional[Path] = None) -> Any:
    """Tải pipeline tiền xử lý"""
    if filepath is None:
        filepath = config.SAVED_MODELS_DIR / "pipeline.pkl"
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy pipeline tại: {filepath}")

    with open(filepath, "rb") as f:
        return pickle.load(f)
