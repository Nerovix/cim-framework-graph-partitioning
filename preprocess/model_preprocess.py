import onnx
import onnxoptimizer
import sys
import os

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
