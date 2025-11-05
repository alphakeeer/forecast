import logging
import os
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

def setup_logger(log_path: str | None = None, level=logging.INFO):
    """
    初始化全局 logger。
    - 若传入 log_path，则日志会同时写入文件。
    - 若 log_path 为 None，则只输出到控制台。
    - 自动避免 tqdm 被中断。
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    # 通用日志格式
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 若传入 log_path，则启用文件输出
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # tqdm 兼容 —— 关键！
    # 让 logging 输出自动避开 tqdm 的进度条区域
    logging.getLogger().addHandler(TqdmCompatibleHandler())

    return logger


class TqdmCompatibleHandler(logging.StreamHandler):
    """
    使 logging 输出与 tqdm 共存，不会打断进度条。
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # 关键：使用 tqdm.write()
            self.flush()
        except Exception:
            self.handleError(record)
