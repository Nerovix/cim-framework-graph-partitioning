import onnx
import onnxoptimizer
import sys
import os


def simplify_model(model_path, model_name, input_shape=(1, 3, 32, 32)):
    simplified_path = f'data/simplified_model_files/{model_name}-simplified.onnx'
    if os.path.exists(simplified_path):
        print(f"Model {simplified_path} already exists.")
        return simplified_path

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

        if not os.path.exists('data/simplified_model_files'):
            os.makedirs('data/simplified_model_files', exist_ok=True)
        onnx.save(loaded_model, simplified_path)
        return simplified_path

    sys.exit(f'failed to open onnx file {model_path}')
