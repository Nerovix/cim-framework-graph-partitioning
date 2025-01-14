import main
import cimpara
import os

output_dir = './instruction_files'
os.makedirs(output_dir, exist_ok=True)

for model_name in ['mobilenet', 'resnet18', 'vgg19', 'efficientnet']:
    for T in [4, 8, 12, 16]:
        for B in [8, 16]:
            for partition_mode in [3, 4, 5]:
                cimpara.onnx_file_path = f'./model_files/{model_name}-simplified.onnx'
                if os.path.exists(cimpara.onnx_file_path) == False:
                    continue
                cimpara.T = T
                cimpara.B = B
                cimpara.partition_mode = partition_mode
                strategy = 'dp'
                if partition_mode == 1:
                    strategy = 'baseline1'
                elif partition_mode == 2:
                    strategy = 'baseline2'
                elif partition_mode == 3:
                    strategy = '2x_communication_time'
                elif partition_mode == 4:
                    strategy = 'sum_calc_time'
                elif partition_mode == 5:
                    strategy = '0.5x_load_time'

                cimpara.instructions_file_path = f'{output_dir}/instructions_{
                    model_name}_{strategy}_T{T}_B{B}.json'
                main.main()

# os.system('zip instruction_files.zip instruction_files_ -r')
