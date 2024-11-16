import main
import cimpara
import os

for model_name in ['mobilenet', 'resnet18', 'vgg19', 'efficientnet']:
    for T in [4, 8, 12, 16]:
        for B in [8, 16]:
            for partition_mode in [0, 1, 2]:
                cimpara.onnx_file_path = f'./model_files/{ \
                    model_name}-simplified.onnx'
                cimpara.T = T
                cimpara.B = B
                cimpara.partition_mode = partition_mode
                strategy = 'dp'
                if partition_mode == 1:
                    strategy = 'baseline1'
                elif partition_mode == 2:
                    strategy = 'baseline2'
                cimpara.instructions_file_path = f'./instruction_files/instructions_{ \
                    model_name}_{strategy}_T{T}_B{B}.json'
                main.main()
os.system('zip instruction_files.zip instruction_files -r')
