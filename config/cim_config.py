import os
import importlib.util
import utils.pattern_maps_gen as pm_gen

onnx_file_path = "../data/model_files/efficientnet-simplified.onnx"

instructions_file_path = 'instructions.json'

# ----------------config--------------------
m = 16 # element rows
n = 8 # element columns
H = 32 # macro rows (# element wise)
W = 8 # macro columns (# element wise)
T = 4 # MG size
K = 16 # MG num
local_memory_size = 512
C = 64  # number of cores
P = 8  # cores per row
Q = 8  # cores per column
B = 8  # noc bandwidth
global_memory_bandwidth = 64
batch_size = 8
weight_width = 8  # width of weights
feature_width = 8  # width of activation values

partition_mode = 0

def channels_on_a_core():
    return n * W // weight_width * T


pattern_maps = []
pattern_pos_lists = []

def update_pos_lists():
    num = C
    global pattern_maps, pattern_pos_lists
    pattern_maps = []
    pattern_pos_lists.clear()
    pm_path = os.path.join(os.path.dirname(__file__), f'../data/pattern_maps/pm{num}.py')
    if not os.path.exists(pm_path):
        # create the file if it doesn't exist, using the functions in pattern_maps.py file
        pm_gen.make_pattern_maps(n_cores=num, pm_path=pm_path)

    spec = importlib.util.spec_from_file_location(f'patterns.pm{num}', pm_path)
    
    pm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm_module)
    pattern_maps = getattr(pm_module, f'pattern_maps_{num}')

    for pattern_map in pattern_maps:
        pos_lists = [(-1, -1)] * num
        for x in range(len(pattern_map)):
            for y in range(len(pattern_map[x])):
                assert pos_lists[pattern_map[x][y]] == (-1, -1), "replicate position r u kidding"
                pos_lists[pattern_map[x][y]] = (x, y)
        pattern_pos_lists.append(pos_lists)