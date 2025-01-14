import onnx
import onnxoptimizer


def simplify_model(model_path):
    
    with open(model_path, "rb") as f:
        loaded_model = onnx.load(f)

        # simplify the model
        passes = ["eliminate_identity", "eliminate_deadend"]
        loaded_model = onnxoptimizer.optimize(loaded_model, passes) 
        onnx.save(loaded_model, "./model_files/mobilenet-simplified.onnx")