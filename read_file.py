import onnx
from onnx import shape_inference
from onnx import GraphProto, ModelProto, NodeProto

def load_onnx_model(file_path):
    model = onnx.load(file_path)
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph
    return graph

def print_graph_nodes(graph):
    for i,node in enumerate(graph.node):
        print(f"{i}, {node.name}")
        '''
        print(f"Node name: {node.name}")
        print(f"Node op_type: {node.op_type}")
        print("Node inputs:", node.input)
        print("Node outputs:", node.output)
        print()
        '''

def get_tensor_shape(onnx_graph, tensor_name):
    
    for initializer in onnx_graph.initializer:
        if initializer.name == tensor_name:
            shape = [d.dim_value for d in initializer.dims]
            return shape
    
    for input_info in onnx_graph.input:
        if input_info.name == tensor_name:
            shape = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
            return shape
    
    for output_info in onnx_graph.output:
        if output_info.name == tensor_name:
            shape = [d.dim_value for d in output_info.type.tensor_type.shape.dim]
            return shape
    
    for value_info in onnx_graph.value_info:
        if value_info.name == tensor_name:
            shape = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
            return shape
    
    raise ValueError(f"Tensor with name '{tensor_name}' not found in the model.")