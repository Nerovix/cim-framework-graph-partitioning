import logging
import os
from datetime import datetime

# Get the absolute path of the directory where the script is located
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create the full path for the logs directory
log_dir = os.path.join(project_root, "data", "logs")

logger = logging.getLogger('graph_partitioning')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime("CG_log_%Y-%m-%d_%H-%M-%S.log")
# Create the full path for the log file
log_filepath = os.path.join(log_dir, log_filename)
file_handler = logging.FileHandler(log_filepath, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
