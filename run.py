import math
import os
import sys
import argparse
import main
import config.cim_config as c_conf
from preprocess.model_preprocess import simplify_model

parser = argparse.ArgumentParser()
parser.add_argument('-T', type=int, default=8, help="T in {4, 8, 12, 16}")
parser.add_argument('-K', type=int, default=16, help="K in {4, 8, 16}")
parser.add_argument('-B', type=int, default=16, help="B in {8, 16}")
parser.add_argument('-C', type=int, default=64, help="C in {1, 64, 144}")
parser.add_argument('--batch-size', type=int, default=8, help="batch size")
parser.add_argument('--model-path', type=str, required=True, help="onnx model file path, e.g. data/model_files/resnet18.onnx")
parser.add_argument('--strategy', type=str, default="dp", help="strategy in {dp, baseline1, baseline2, 2x_communication_time, sum_calc_time, 0.5x_load_time, pipelined_calculate_time}")
parser.add_argument('--output_dir', type=str, default="data/instruction_files", help="output directory for the instruction files")
parser.add_argument('--visualize', action='store_true', help="whether to visualize the partitioning result")

args = parser.parse_args()

if args.T not in [4, 8, 12, 16]:
    sys.exit("T should be in {4, 8, 12, 16}")
if args.K not in [4, 8, 16]:
    sys.exit("K should be in {4, 8, 16}")
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

c_conf.onnx_file_path = simplified_path
if not os.path.exists(c_conf.onnx_file_path):
    sys.exit(f"onnx file doesn't exist: {c_conf.onnx_file_path}")

c_conf.T = args.T
c_conf.K = args.K
c_conf.B = args.B
c_conf.C = args.C
c_conf.batch_size = args.batch_size
c_conf.P = int(math.sqrt(args.C))
c_conf.Q = int(c_conf.P)
c_conf.partition_mode = partition_mode
c_conf.update_pos_lists()
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

c_conf.instructions_file_path = f'{output_dir}/instructions_{model_name}_{args.strategy}_T{args.T}_B{args.B}_C{args.C}_batch{c_conf.batch_size}.json'
c_conf.visualize_flag = args.visualize

main.main()
