#SBATCH -p selene
#SBATCH -A sa
#SBATCH --job-name=sa-tts:milk
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --dependency=singleton

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

MNT_USER_ROOT=/lustre/fsw/sa/hryu
MNT_DATA_ROOT=/lustre/fsw/sa/hryu/datasets
USER_ROOT=/scratch
DATA_ROOT=/dataset
SRC_DIR=$USER_ROOT/MILK

TASK=ex01
TASK_DIR = $USER_ROOT/$TASK   
SAMPLE_DIR=$TASK_DIR/sample
OUTPUT_DIR=$TASK_DIR/output
 
TRAIN_CMD="cd $TASK_DIR && \
        python -Wd -m torch.distributed.alunch --nproc_per_node 8 train.py  \
        --cuda \
        -o $OUTPUT_DIR \
        --sample-dir $SAMPLE_DIR \
        --epoches 1500 \
        --epochs-per-checkpoint 10 \
        --warmup-steps 1000 \
        -lr 0.1 \
        --fp16 \
        --amp pytorch \
        -bs 256 | 1>&2 | tee log_$DATETIME.txt "

srun -l \
     --container-image=hryu/pytorch:tts01 \
     --container-mounts=$MNT_USER_ROOT:$USER_ROOT,$MNT_DATA_ROOT:$DATA_ROOT   \
     --output=$TASK_DIR/%x_%j_$DATETIME.log sh -c "${TRAIN_CMD}"
set +x   
