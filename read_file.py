import onnx
from onnx import shape_inference
from logging_config import logger

# mem = dict()


def load_onnx_model(file_path):
    model = onnx.load(file_path)
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph
    # mem=dict()
    return graph


def print_graph_nodes(graph):
    logger.info(f"Number of nodes: {len(graph.node)}")
    for i, node in enumerate(graph.node):
        logger.debug(f"Node id:{i}, Node Name:{node.name}, Node op_type:{node.op_type}")


def get_tensor_shape(onnx_graph, tensor_name):
    # if tensor_name in mem:
    # return [_ for _ in mem[tensor_name]];
    for initializer in onnx_graph.initializer:
        if initializer.name == tensor_name:
            shape = [d if type(d) is int else d.dim_value for d in initializer.dims]
            # mem[tensor_name] = shape
            return [_ for _ in shape]

    for input_info in onnx_graph.input:
        if input_info.name == tensor_name:
            shape = [d if type(d) is int else d.dim_value for d in input_info.type.tensor_type.shape.dim]
            # mem[tensor_name] = shape
            return [_ for _ in shape]

    for output_info in onnx_graph.output:
        if output_info.name == tensor_name:
            shape = [
                d if type(d) is int else d.dim_value for d in output_info.type.tensor_type.shape.dim]
            # mem[tensor_name] = shape
            return [_ for _ in shape]

    for value_info in onnx_graph.value_info:
        if value_info.name == tensor_name:
            shape = [d if type(d) is int else d.dim_value for d in value_info.type.tensor_type.shape.dim]
            # mem[tensor_name] = shape
            return [_ for _ in shape]

    raise ValueError(f"Tensor with name '{tensor_name}' not found in the model.")
