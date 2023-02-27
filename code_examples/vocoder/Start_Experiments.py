import os
from datetime import datetime

iterations = 2
Module_Names = ['DenseNet_3D', 'ResNet_3D'] 
Model_Names = ['DenseNet264_3D', 'ResNet152_3D']
Model_Dir = ''
Epochs = 5
Batch_Size = 2

Experiments_Time = datetime.now().strftime('%y%m%d_%H%M%S')
for iteration in range(1, iterations+1):
    for i, (module_name, model_name) in enumerate(zip(Module_Names, Model_Names)):
        os.system(
    f'''
    python\
        train.py\
        -et={Experiments_Time}\
        --seed={iteration}\
        -mun={module_name}\
        -men={model_name}\
        -medir={Model_Dir}\
        -od={f'output_{Experiments_Time}/{model_name}_iter_{iteration}'}
        -bs={Batch_Size}\
        -ep={Epochs}\
    '''
        )