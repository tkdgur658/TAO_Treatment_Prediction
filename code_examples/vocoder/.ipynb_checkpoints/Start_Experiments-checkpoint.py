import os
from datetime import datetime

iterations = 2
module_names = ['DenseNet_3D', 'ResNet_3D'] 
model_names = ['DenseNet264_3D', 'ResNet152_3D']
model_dir = 'Models'
epochs = 5
batch_size = 2

Experiments_Time = datetime.now().strftime('%y%m%d_%H%M%S')
for iteration in range(1, iterations+1):
    for i, (module_name, model_name) in enumerate(zip(module_names, model_names)):
        os.system(
    f'''
    python\
        train.py\
        -et={Experiments_Time}\
        --seed={iteration}\
        -mun={module_name}\
        -men={model_name}\
        -medir={model_dir}\
        -od={f'output_{Experiments_Time}/{model_name}_iter_{iteration}'}\
        -bs={batch_size}\
        -ep={epochs}\
    '''
        )