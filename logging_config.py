import logging

logger = logging.getLogger('graph_partitioning')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

file_handler = logging.FileHandler(
    'cim_framework_graph_partitioning.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
