import onnx
import onnxoptimizer
import sys
import os


def simplify_model(model_path, model_name):

    with open(model_path, "rb") as f:
        loaded_model = onnx.load(f)

        # simplify the model
        passes = ["eliminate_identity", "eliminate_deadend"]
        loaded_model = onnxoptimizer.optimize(loaded_model, passes)
        simplified_path = f'./simplified_model_files/{model_name}-simplified.onnx'
        os.makedirs('./simplified_model_files', exist_ok=True)
        onnx.save(loaded_model, simplified_path)
        return simplified_path

    sys.exit(f'failed to open onnx file {model_path}')
