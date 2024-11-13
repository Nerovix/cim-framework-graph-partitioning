import json
from read_file import load_onnx_model, print_graph_nodes
from process import process
import sys
sys.setrecursionlimit(100000)

if __name__ == "__main__":

    # cores_needed=[10,6,9,4,5,3]
    # allo=put_nodes_on_chip(cores_needed)
    # # print(allo)
    # print (cores_needed)
    # print_allocation(allo)

    model_name = ''

    onnx_file_path = "./model_files/mobilenetv2-12-qdq.onnx"
    # [['0-1-6', '1-1-6', '2-2-7', '3-1-2', '4-3-2', '5-1-2', '6-3-2', '7-1-1', '8-3-1', '9-1-1', '10-3-1', '11-1-1', '12-3-1', '13-1-1', '14-6-1', '15-1-1'],
    # ['16-6-8', '17-1-8'], ['18-6-4', '19-1-6', '20-6-4', '21-2-5'], ['22-9-4', '23-2-8'], 
    # ['24-9-2', '25-2-6', '26-9-3', '27-3-2'], ['28-15-1', '29-3-3', '30-15-2', '31-3-3'], ['32-15-2', '33-5-6'], ['34-20-3']]

    # onnx_file_path = "./model_files/resnet50-v1-12-qdq.onnx"
    # [['0-1-8', '1-1-8', '2-1-8'], ['3-4-4', '4-4-4', '5-1-4', '6-1-8', '7-4-4', '8-1-4'], ['9-1-8', '10-4-8', '11-2-4', '12-2-8'],
    # ['13-8-3', '14-8-4', '15-2-4'], ['16-2-8'], ['17-8-4', '18-2-8', '19-2-8'], ['20-8-4', '21-2-8', '22-2-8'], ['23-8-4', '24-4-2', '25-4-6'],
    # ['26-16-1', '27-16-2', '28-4-4'], ['29-4-8'], ['30-16-2', '31-4-8'], ['32-4-8'], ['33-16-2', '34-4-8'], ['35-4-8'],
    # ['36-16-2', '37-4-8'], ['38-4-8'], ['39-16-2', '40-4-8'], ['41-4-8'], ['42-16-3', '43-8-2'], ['44-8-8'], ['45-32-2'],
    # ['46-32-2'], ['47-8-8'], ['48-8-8'], ['49-32-1', '50-8-4'], ['51-8-8'], ['52-32-2']]

    # onnx_file_path = "./model_files/resnet18-v1-7.onnx"
    # [['0-1-8', '1-1-8', '2-1-8', '3-1-8', '4-1-8', '6-2-8'], 
    # ['5-2-2', '7-2-4', '8-2-4', '9-2-4', '11-4-2', '10-4-1', '12-4-2', '13-4-2', '14-4-2'], 
    # ['16-8-2', '15-8-1', '17-8-5'], ['18-8-4', '19-8-4']]

    # onnx_file_path = "./model_files/vgg16-12-qdq.onnx"
    # [['0-1-4', '1-1-8'], ['2-2-8', '3-2-8', '4-4-8'], ['5-4-8', '6-4-8'], ['7-8-8'], ['8-8-4', '9-8-4'], ['10-8-8'], ['11-8-4', '12-8-4']]

    # onnx_file_path = "./model_files/efficientnet-lite4-11-qdq.onnx"
    # [['0-1-8', '1-1-8', '2-3-8', '3-1-3', '4-3-2', '5-1-3', '6-3-3', '7-1-3'], ['8-3-8', '9-1-8', '10-3-8', '11-1-8'], 
    # ['12-6-8', '13-1-8'], ['14-6-8', '15-1-8'], ['16-6-8', '17-1-8'], ['18-6-8', '19-2-8'], ['20-11-4', '21-2-8'], 
    # ['22-11-4', '23-2-8'], ['24-11-4', '25-2-8'], ['26-11-4', '27-2-8'], ['28-11-4', '29-2-8'], ['30-11-3', '31-3-8'], 
    # ['32-15-3', '33-3-6'], ['34-15-1', '35-3-3', '36-15-2', '37-3-3'], ['38-15-1', '39-3-3', '40-15-2', '41-3-3'],
    # ['42-15-3', '43-5-3'], ['44-26-1', '45-5-7'], ['46-26-1', '47-5-7'], ['48-26-1', '49-5-7'], ['50-26-1', '51-5-7'],
    # ['52-26-1', '53-5-7'], ['54-26-1', '55-5-7'], ['56-26-1', '57-5-7'], ['58-26-1', '59-7-5'], ['60-20-3']]

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
    instructions=process(onnx_graph)
    # print(instructions)
    json_instructions=json.dumps(instructions,ensure_ascii=False,indent=4)
    with open('instructions.json','w') as json_file:
        print(json_instructions,file=json_file)

    print('its done its done its done its done its done its done its done its done akkdfalsdfja;sldf')
    
