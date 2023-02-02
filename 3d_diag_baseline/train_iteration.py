import os
from datetime import datetime

iterations = 2
model_names = ['DenseNet264', 'ResNet152']
max_epochs = 5

DATETIME_NOW = datetime.now().strftime('%y%m%d_%H%M%S')
for model_name in model_names:
    for i in range(1, iterations+1):
        os.system(
    f'''
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    CUDA_VISIBLE_DEVICES=0,1 \
    OMP_NUM_THREADS=2 \
    torchrun\
        --nnodes=1\
        --nproc_per_node=2\
        --node_rank=0\
        --master_addr="localhost"\
        --master_port=12355\
        train.py\
        -et={DATETIME_NOW}\
        -o='experiment_result.csv'\
        -ep={max_epochs}\
        --iteration={i}\
        --seed={i}\
        --model_name={model_name}\
        --reproducibility\
    '''
        )