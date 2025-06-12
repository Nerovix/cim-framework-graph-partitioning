import json
from preprocess.read_file import load_onnx_model
from preprocess.read_file import print_graph_nodes
from optimization.cg_mapping import cg_mapping
from config.logger_config import logger
import config.cim_config as c_conf
import sys
sys.setrecursionlimit(100000)  # for dfs


def main():

    logger.info(
        f'running with onnx_file_path = {c_conf.onnx_file_path},T = {c_conf.T},B = {c_conf.B},partition_mode = {c_conf.partition_mode}'
    )
    model = load_onnx_model(c_conf.onnx_file_path)

    print_graph_nodes(model.graph)

    instructions = cg_mapping(model)

    # Output instructions in json format
    logger.info('Output instructions in json format...')
    json_instructions = json.dumps(instructions, ensure_ascii=False, indent=4)
    with open(c_conf.instructions_file_path, 'w') as json_file:
        print(json_instructions, file=json_file)
    logger.info('Output instructions in json format completed.')
    logger.info('Graph-partitioning completed.')


if __name__ == "__main__":
    main()
