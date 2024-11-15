import json
from read_file import load_onnx_model, print_graph_nodes
from process import process
import sys
sys.setrecursionlimit(100000)

if __name__ == "__main__":
    
    # 0: dp
    # 1: 先尽可能往上放，然后再复制
    # 2: 不复制，一直往上放，直到放不动。纯纯的比1还蠢
    partition_mode=0 

    
    model_name = ''

    onnx_file_path = "./model_files/mobilenet-simplified.onnx"

    onnx_file_path = "./model_files/resnet18-simplified.onnx"
    
    onnx_file_path = "./model_files/vgg19-simplified.onnx"
    
    onnx_file_path = "./model_files/efficientnet-simplified.onnx"
    
    onnx_graph = load_onnx_model(onnx_file_path)

    if model_name == '':
        if onnx_file_path.lower().find('mobilenet') != -1:
            model_name = 'mobilenet'
        elif onnx_file_path.lower().find('resnet') != -1:
            model_name = 'resnet'
        elif onnx_file_path.lower().find('vgg') != -1:
            model_name = 'vgg'
        elif onnx_file_path.lower().find('efficientnet') != -1:
            model_name = 'efficientnet'
    assert model_name != '', 'Model name is required.'


    print_graph_nodes(onnx_graph)

    # if model_name=='alexnet' or model_name=='vgg' or model_name=='resnet':
    instructions = process(onnx_graph,partition_mode)
    # print(instructions)
# '''
    json_instructions = json.dumps(instructions, ensure_ascii=False, indent=4)
    with open('instructions.json', 'w') as json_file:
        print(json_instructions, file=json_file)

    print('its done its done its done its done its done its done its done its done akkdfalsdfja;sldf')
# '''