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

    onnx_file_path = "./model_files/mobilenetv2-12-qdq.onnx"
    # [0-1-7, 1-1-7, 2-2-7, 3-1-2, 4-3-2, 5-1-2, 6-3-2, 7-1-1, 8-3-1, 9-1-1, 10-3-1, 11-1-1, 12-3-1, 13-1-1, ]
    # [14-6-1, 15-1-1, 16-6-1, 17-1-1, 18-6-1, 19-1-1, 20-6-1, 21-2-1, 22-9-1, 23-2-1, 24-9-1, 25-2-1, 26-9-1, 27-3-1, ]
    # [28-15-1, 29-3-1, 30-15-1, 31-3-1, 32-15-1, 33-5-1, ]
    # [34-20-1, ]

    onnx_file_path = "./model_files/resnet50-v1-12-qdq.onnx"
    # [0-1-8, 1-1-2, 2-1-4, 4-4-2, 5-1-2, 6-1-4, 7-4-2, 8-1-2, 9-1-4, 10-4-2, 11-2-1, 12-2-2, 3-4-2, ]
    # [13-8-1, 14-8-1, 15-2-1, 16-2-1, 17-8-1, 18-2-1, 19-2-1, 20-8-1, 21-2-1, 22-2-1, 23-8-1, 24-4-1, 25-4-1, ]
    # [26-16-1, 27-16-1, 28-4-1, 29-4-1, 30-16-1, 31-4-1, 32-4-1, ]
    # [33-16-1, 34-4-1, 35-4-1, 36-16-1, 37-4-1, 38-4-1, ]
    # [39-16-1, 40-4-1, 41-4-1, 42-16-1, 43-8-1, 44-8-1, ]
    # [45-32-1, 46-32-1, ]
    # [47-8-1, 48-8-1, 49-32-1, 50-8-1, 51-8-1, ]
    # [52-32-1, ]
    
    onnx_file_path = "./model_files/resnet18-v1-7.onnx"
    # [0-1-2, 1-1-1, 2-1-1, 3-1-1, 4-1-1, 6-2-1, 5-2-1, 7-2-1, 8-2-1, 9-2-1, 11-4-1, 10-4-1, 12-4-1, 13-4-1, 14-4-1, 16-8-1, 15-8-1, 17-8-1, ]
    # [18-8-1, 19-8-1, ]

    onnx_file_path = "./model_files/vgg16-12-qdq.onnx"
    # [0-1-5, 1-1-8, 2-2-3, 3-2-4, 4-4-2, 5-4-2, 6-4-2, 7-8-1, ]
    # [8-8-2, 9-8-2, 10-8-1, 11-8-1, 12-8-1, ]

    # onnx_file_path = "./model_files/efficientnet-lite4-11-qdq.onnx"
    # [0-1-8, 1-1-8, 2-3-8, 3-1-3, 4-3-2, 5-1-3, 6-3-3, 7-1-3, ]
    # [8-3-8, 9-1-8, 10-3-8, 11-1-4, ]
    # [12-6-1, 13-1-2, 14-6-1, 15-1-2, 16-6-1, 17-1-2, 18-6-2, 19-2-1, 20-11-1, 21-2-1, 22-11-1, 23-2-1, ]
    # [24-11-1, 25-2-2, 26-11-1, 27-2-2, 28-11-1, 29-2-2, 30-11-1, 31-3-2, ]
    # [32-15-1, 33-3-2, 34-15-1, 35-3-2, 36-15-1, 37-3-2, ]
    # [38-15-1, 39-3-2, 40-15-1, 41-3-2, 42-15-1, 43-5-1, ]
    # [44-26-1, 45-5-1, 46-26-1, 47-5-1, ]
    # [48-26-1, 49-5-1, 50-26-1, 51-5-1, ]
    # [52-26-1, 53-5-1, 54-26-1, 55-5-1, ]
    # [56-26-1, 57-5-1, 58-26-1, 59-7-1, ]
    # [60-20-1, ]

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