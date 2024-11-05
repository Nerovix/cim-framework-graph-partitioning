from read_file import load_onnx_model,print_graph_nodes
from process_alexnet_vgg import process_alexnet_vgg
import sys
sys.setrecursionlimit(100000)

if __name__ == "__main__":
    
    # cores_needed=[10,6,9,4,5,3]
    # allo=put_nodes_on_chip(cores_needed)
    # # print(allo)
    # print (cores_needed)
    # print_allocation(allo)
    
    
    
    model_name=''

    # onnx_file_path = "./model_files/bvlcalexnet-12-qdq.onnx"
    # onnx_file_path = "./model_files/mobilenetv2-12-qdq.onnx"
    # onnx_file_path = "./model_files/resnet50-v1-12-qdq.onnx"
    # onnx_file_path = "./model_files/resnet18-v1-7.onnx"
    # onnx_file_path = "./model_files/vgg16-12-qdq.onnx"
    onnx_file_path = "./model_files/efficientnet-lite4-11-qdq.onnx"
    
    
    
    onnx_graph = load_onnx_model(onnx_file_path)
    
    
    if model_name=='':
        if onnx_file_path.lower().find('alexnet')!=-1:
            model_name='alexnet'
        elif onnx_file_path.lower().find('mobilenet')!=-1:
            model_name='mobilenet'
        elif onnx_file_path.lower().find('resnet')!=-1:
            model_name='resnet'
        elif onnx_file_path.lower().find('vgg')!=-1:
            model_name='vgg'
        elif onnx_file_path.lower().find('efficientnet')!=-1:
            model_name='efficientnet'
    assert model_name != '', 'Model name is required.'
    
    
    print_graph_nodes(onnx_graph)

    # if model_name=='alexnet' or model_name=='vgg' or model_name=='resnet':
    process_alexnet_vgg(onnx_graph)
        

        

