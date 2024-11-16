import json
from read_file import load_onnx_model
from process import process
import cimpara as cp
import sys
sys.setrecursionlimit(100000)

def main():
    print(f'running with onnx_file_path = {cp.onnx_file_path}, T = {cp.T}, B = {cp.B}, partition_mode = {cp.partition_mode}')

    onnx_graph = load_onnx_model(cp.onnx_file_path)

    # print_graph_nodes(onnx_graph)

    instructions = process(onnx_graph)
    # print(instructions)
    # '''
    json_instructions = json.dumps(instructions, ensure_ascii=False, indent=4)
    with open(cp.instructions_file_path, 'w') as json_file:
        print(json_instructions, file=json_file)

    print('its done its done its done its done its done its done its done its done akkdfalsdfja;sldf')
# '''
    

if __name__ == "__main__":
    main()