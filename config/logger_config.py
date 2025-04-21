import logging
import os
from datetime import datetime

logger = logging.getLogger('graph_partitioning')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

os.makedirs('data/logs/', exist_ok=True)
log_filename = datetime.now().strftime("CG_log_%Y-%m-%d_%H-%M-%S.log")
file_handler = logging.FileHandler("data/logs/" + log_filename, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
