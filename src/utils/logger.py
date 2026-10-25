"""
logger.py - Hệ thống Logging và Tiện ích Đo đạc Thời gian Thực thi
===================================================================
- setup_logger(name, log_file, level): Cấu hình logging đa kênh (Console + File UTF-8)
- log_step(step_name): Decorator đo thời gian thực thi (runtime) từng bước
- print_section_header(title, step_num): In tiêu đề định dạng trực quan cho terminal
"""
import sys
import time
import logging
from pathlib import Path
from functools import wraps
from typing import Optional, Callable, Any

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src import config


def setup_logger(
    name: str = "sales_predictor",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Cấu hình và trả về Logger chuẩn cho toàn bộ quy trình dự án.
    Hỗ trợ xuất log song song ra Console và tệp nhật ký UTF-8.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh duplicate handlers khi gọi nhiều lần
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (ghi file nhật ký nếu có)
    target_log_file = log_file or getattr(config, "LOG_FILE", ROOT_DIR / "logs" / "pipeline.log")
    try:
        target_log_file = Path(target_log_file)
        target_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target_log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Không thể khởi tạo FileHandler tại {target_log_file}: {e}")

    return logger


def log_step(step_name: Optional[str] = None) -> Callable:
    """
    Decorator đo thời gian chạy của hàm và ghi log thời lượng thực thi.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            action_name = step_name or func.__name__.replace("_", " ").title()
            logger = logging.getLogger("sales_predictor")
            logger.info(f"⏳ Bắt đầu thực hiện: [{action_name}]...")
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - t0
                logger.info(f"✓ Hoàn tất [{action_name}] trong {elapsed:.2f} giây.")
                return result
            except Exception as e:
                elapsed = time.time() - t0
                logger.error(f"❌ Thất bại tại [{action_name}] sau {elapsed:.2f}s: {e}")
                raise
        return wrapper
    return decorator


def print_section_header(title: str, step_num: Optional[int] = None) -> None:
    """
    In thanh ngăn cách và tiêu đề giai đoạn đẹp mắt ra console.
    """
    width = 80
    header_title = f" BƯỚC {step_num}: {title.upper()} " if step_num is not None else f" {title.upper()} "
    print("\n" + "=" * width)
    print(header_title.center(width, "★"))
    print("=" * width + "\n")
