import os
import sys
import argparse
import main
import cimpara
from model_simplify import simplify_model

output_dir = './instruction_files'
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-T', type=int, default=4, help="T in {4, 8, 12, 16}")
parser.add_argument('-B', type=int, default=8, help="B in {8, 16}")
parser.add_argument('--model-path', type=str, required=True, help="onnx model file path, e.g. ./model_files/resnet18.onnx")
parser.add_argument('--strategy', type=str, default="dp", help="strategy in {dp, baseline1, baseline2, 2x_communication_time, sum_calc_time, 0.5x_load_time, pipelined_calculate_time}")

args = parser.parse_args()

if args.T not in [4, 8, 12, 16]:
    sys.exit("T should be in {4, 8, 12, 16}")
if args.B not in [8, 16]:
    sys.exit("B should be in {8, 16}")

if not os.path.isfile(args.model_path):
    sys.exit(f"Model file doesn't exist: {args.model_path}")

model_name = os.path.splitext(os.path.basename(args.model_path))[0]

simplified_path = simplify_model(args.model_path, model_name)

allowed_strategies = ['dp', 'baseline1', 'baseline2', '2x_communication_time', 'sum_calc_time', '0.5x_load_time', 'pipelined_calculate_time']
if args.strategy not in allowed_strategies:
    sys.exit("strategy is illegal")

print(args.model_path,file=sys.stderr)
print(model_name,file=sys.stderr)
print(simplified_path,file=sys.stderr)

if args.strategy == 'baseline1':
    partition_mode = 1
elif args.strategy == 'baseline2':
    partition_mode = 2
elif args.strategy == '2x_communication_time':
    partition_mode = 3
elif args.strategy == 'sum_calc_time':
    partition_mode = 4
elif args.strategy == '0.5x_load_time':
    partition_mode = 5
elif args.strategy == 'pipelined_calculate_time':
    partition_mode = 6
elif args.strategy == 'dp':
    partition_mode = 0
else:
    partition_mode = 0

cimpara.onnx_file_path = simplified_path
if not os.path.exists(cimpara.onnx_file_path):
    sys.exit(f"onnx file doesn't exist: {cimpara.onnx_file_path}")

cimpara.T = args.T
cimpara.B = args.B
cimpara.partition_mode = partition_mode
cimpara.instructions_file_path = f'{output_dir}/instructions_{model_name}_{args.strategy}_T{args.T}_B{args.B}.json'

main.main()
