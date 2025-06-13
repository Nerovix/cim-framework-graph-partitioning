import onnx
import onnxoptimizer
import sys
import os
import math
import config.cim_config as c_conf
import numpy as np
from onnx import helper, numpy_helper
from config.logger_config import logger

def simplify_model(model_path, model_name, input_shape=(1, 3, 32, 32)):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simplified_model_path = project_root + f'/data/simplified_model_files/{model_name}-simplified.onnx'
    if os.path.exists(simplified_model_path):
        print(f"Model {simplified_model_path} already exists.")
        return simplified_model_path

    with open(model_path, "rb") as f:
        loaded_model = onnx.load(f)

        # simplify the model
        passes = ["eliminate_identity", "eliminate_deadend"]
        loaded_model = onnxoptimizer.optimize(loaded_model, passes)
        

        input_tensor = loaded_model.graph.input[0]  # acquire the first input
        # update the input shape
        for i, dim_size in enumerate(input_shape):
            input_tensor.type.tensor_type.shape.dim[i].dim_value = dim_size

        loaded_model = onnx.shape_inference.infer_shapes(loaded_model, data_prop=True)

        onnx.checker.check_model(loaded_model)

        simplified_path = project_root + '/data/simplified_model_files'
        if not os.path.exists(path = simplified_path):
            os.makedirs(simplified_path, exist_ok=True)
        onnx.save(loaded_model, simplified_model_path)
        return simplified_model_path

    sys.exit(f'failed to open onnx file {model_path}')


def onnx_split_large_conv_pass(orig_model: onnx.ModelProto):
    onnx_graph = orig_model.graph
    threshold = c_conf.H * c_conf.m * c_conf.K
    new_nodes = []
    new_inits = []

    for node in onnx_graph.node:
        if node.op_type != "Conv":
            new_nodes.append(node)
            continue

        weight_name = node.input[1]
        w_init = next((t for t in onnx_graph.initializer if t.name == weight_name), None)
        if w_init is None:
            new_nodes.append(node)
            continue

        W = numpy_helper.to_array(w_init)  # shape [C_out, C_in, Kh, Kw]
        Cout, Cin, Kh, Kw = W.shape
        total = Cin * Kh * Kw
        if total <= threshold:
            new_nodes.append(node)
            continue

        split_num = math.ceil(total / threshold)
        base = Cin // split_num
        rem = Cin % split_num
        splits = [(base + 1) if i < rem else base for i in range(split_num)]

        slice_outputs = []
        c_in_offset = 0
        if len(splits)>1:
            logger.info('' + f'Conv node {node.name} has too large kernel size, splitting into {len(splits)} parts.')
        for i, c_in_chunk in enumerate(splits):
            start = c_in_offset
            end = start + c_in_chunk

            # make starts, ends, axes, steps tensors for Slice
            starts_name = f"{node.name}_starts_{i}"
            ends_name   = f"{node.name}_ends_{i}"
            axes_name   = f"{node.name}_axes_{i}"
            steps_name  = f"{node.name}_steps_{i}"

            new_inits += [
                numpy_helper.from_array(np.array([start], dtype=np.int64), name=starts_name),
                numpy_helper.from_array(np.array([end],   dtype=np.int64), name=ends_name),
                numpy_helper.from_array(np.array([1],     dtype=np.int64), name=axes_name),
                numpy_helper.from_array(np.array([1],     dtype=np.int64), name=steps_name),
            ]

            slice_out = f"{node.name}_slice_{i}"
            slice_node = helper.make_node(
                "Slice",
                inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
                outputs=[slice_out],
                name=f"{node.name}_Slice_part{i}"
            )
            new_nodes.append(slice_node)

            # weight chunk
            W_chunk = W[:, start:end, :, :].copy()
            w_chunk_name = f"{weight_name}_split_{i}"
            w_chunk_init = numpy_helper.from_array(W_chunk, name=w_chunk_name)
            new_inits.append(w_chunk_init)

            # make Conv for this chunk, preserve attrs
            conv_out = f"{node.output[0]}_part_{i}"
            conv_node = helper.make_node(
                "Conv",
                inputs=[slice_out, w_chunk_name] + (node.input[2:] if len(node.input) > 2 else []),
                outputs=[conv_out],
                name=f"{node.name}_Conv_part{i}",
                domain=node.domain
            )
            # copy attributes
            conv_node.attribute.extend(node.attribute)
            new_nodes.append(conv_node)

            slice_outputs.append(conv_out)
            c_in_offset = end

        # build a balanced add tree over slice_outputs
        outputs = slice_outputs.copy()
        level = 0
        # iteratively pairwise add until one output remains
        while len(outputs) > 1:
            next_outputs = []
            for i in range(0, len(outputs), 2):
                if i + 1 < len(outputs):
                    a = outputs[i]
                    b = outputs[i+1]
                    # determine output name: final output uses original node.output[0]
                    is_last = (len(outputs) == 2)
                    out_name = node.output[0] if is_last else f"{node.name}_add_l{level}_{i//2}"
                    add_node = helper.make_node(
                        "Add",
                        inputs=[a, b],
                        outputs=[out_name],
                        name=f"{node.name}_Add_l{level}_{i//2}"
                    )
                    new_nodes.append(add_node)
                    next_outputs.append(out_name)
                else:
                    # odd number, carry forward
                    next_outputs.append(outputs[i])
            outputs = next_outputs
            level += 1

    onnx_graph.ClearField("node")
    onnx_graph.node.extend(new_nodes)
    onnx_graph.initializer.extend(new_inits)
    
    # perform shape inference on the updated graph
    model = helper.make_model(onnx_graph)
    model.opset_import.extend(orig_model.opset_import)
    inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    onnx.checker.check_model(inferred)
    orig_model.graph.CopyFrom(inferred.graph)
    return 
