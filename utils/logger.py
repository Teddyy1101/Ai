import logging
import sys

# ===================== CẤU HÌNH LOGGER =====================
logger = logging.getLogger("DrowsinessApp")
logger.setLevel(logging.INFO)

# Xóa mọi handler cũ nếu có
if logger.hasHandlers():
    logger.handlers.clear()

# Tạo handler chỉ in ra terminal (không ghi file)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Định dạng log
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
console_handler.setFormatter(formatter)

# Thêm handler
logger.addHandler(console_handler)

# ===================== HÀM TIỆN ÍCH =====================
def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)
