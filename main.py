import json
from read_file import load_onnx_model
from read_file import print_graph_nodes
from process import process
from logging_config import logger
import cimpara as cp
import sys
sys.setrecursionlimit(100000)  # for dfs


def main():

    logger.info(
        f'running with onnx_file_path = {cp.onnx_file_path},T = {cp.T},B = {cp.B},partition_mode = {cp.partition_mode}'
    )
    onnx_graph = load_onnx_model(cp.onnx_file_path)

    print_graph_nodes(onnx_graph)

    instructions = process(onnx_graph)

    # Output instructions in json format
    logger.info('Output instructions in json format...')
    json_instructions = json.dumps(instructions, ensure_ascii=False, indent=4)
    with open(cp.instructions_file_path, 'w') as json_file:
        print(json_instructions, file=json_file)
    logger.info('Output instructions in json format completed.')
    logger.info('Graph-partitioning completed.')


if __name__ == "__main__":
    main()
