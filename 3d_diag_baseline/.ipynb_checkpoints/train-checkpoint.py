import argparse
import logging
import os
import random
from datetime import datetime
import sys 

import ignite.distributed as idist
import monai
import numpy as np
import pandas as pd
import torch
from data.custom_dataset import CustomDataset
from ignite.metrics import Loss
from ignite.utils import setup_logger
from monai.data import DataLoader
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (CheckpointLoader, CheckpointSaver,
                            LrScheduleHandler, MetricLogger, StatsHandler,
                            ValidationHandler, from_engine)
from monai.inferers import SimpleInferer
from monai.transforms import Compose, RandRotate
from ptflops import get_model_complexity_info
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from utils.csv_logger import ExperimantResult, csv_logger
from utils.metrics import SaveOutputs, get_performances

MODEL_SAVE_DIR = f'./saved_model'

metric_names = ['acc', 'recall', 'precision', 'loss']
TRAIN_ACC, TRAIN_RECALL, TRAIN_PREC, TRAIN_LOSS\
    = ['train_' + metric_name for metric_name in metric_names]
VAL_ACC, VAL_RECALL, VAL_PREC, VAL_LOSS\
    = ['val_' + metric_name for metric_name in metric_names]
TEST_ACC, TEST_RECALL, TEST_PREC, TEST_LOSS\
    = ['test_' + metric_name for metric_name in metric_names]

def ddp_initialize():

    # initialize the distributed training process, every GPU runs in a process
    # dist.init_process_group(backend='nccl')
    idist.initialize('nccl')
    idist.barrier()


def ddp_finalize():
    
    # dist.destroy_process_group()
    idist.finalize()


def data_preparing(args: argparse.Namespace):
    
    seed = args.seed
    
    df_label = pd.read_csv(args.label_file, header=None, index_col=0)

    images = [os.path.join(args.data_dir, f'{i}.npy') for i in df_label.index.to_list()]
    labels = df_label.to_numpy(dtype=np.int64).flatten()

   # Train/Valid/Test split
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, train_size=args.train_test_split_ratio, random_state=seed,
        stratify=labels if len(labels) * (1 - args.train_test_split_ratio) > 1 else None,
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, train_size=args.train_val_split_ratio, random_state=seed,
        stratify=train_labels if len(train_labels) * (1 - args.train_val_split_ratio) > 1 else None,
    )

    # Define transforms
    train_transforms = Compose([RandRotate(range_x=10, range_y=10, range_z=10)])
    val_transforms = None
    test_transforms = None

    # create a training data loader
    train_ds = CustomDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    val_ds = CustomDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
    test_ds = CustomDataset(image_files=test_images, labels=test_labels, transform=test_transforms)

    # Define samplers
    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds)
    test_sampler = DistributedSampler(test_ds)

    # Define loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        sampler=val_sampler,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, val_loader, test_loader


def train(args: argparse.Namespace):

    ddp_initialize()
    
    iteration = args.iteration
    seed=args.iteration
    
    if idist.get_rank() == 0:
        print(f'Iteration {iteration:02d}')
    DATETIME_NOW=args.experiment_time
    TRAIN_TIME = datetime.now().strftime('%y%m%d_%H%M%S')
    MODEL_SAVE_FILE_NAME = f'{TRAIN_TIME}.pt'

    train_loader, val_loader, test_loader = data_preparing(args)

    # Define model, loss, optimizer and scheduler
    device = torch.device(f'cuda:{idist.get_local_rank()}')
    torch.cuda.set_device(device)
    
    net = None
    
#     exec(f'from models.{args.module_name} import {args.model_name}')
#     net = str_to_class(args.model_name)(in_channels=1, num_classes=2)
    if args.model_name == 'DenseNet264':
        net = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1, out_channels=2)
#         from models.DenseNet_3D import DenseNet264_3D
#         net = DenseNet264_3D(in_channels=1, num_classes=2)
    if args.model_name == 'ResNet152':
#         from models.ResNet_3D import ResNet152_3D
#         net = ResNet152_3D(in_channels=1, num_classes=2)
        net = monai.networks.nets.resnet152(spatial_dims=3, n_input_channels=1, num_classes=2)
    net = net.to(device)
    net = DistributedDataParallel(net, device_ids=[device])

    loss_fn = None
    if args.loss == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss

    opt = None
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = 10, gamma=0.5)

    # Define handlers
    train_handlers, val_handlers, test_handlers = [], [], []

    if idist.get_rank() == 0:
        print('=' * 30)
        print('Training Settings\n')
        print(f'{args=}\n')
        print('num_train:', len(train_loader.dataset))
        print('num_val:', len(val_loader.dataset))
        print('num_test:', len(test_loader.dataset))
        print('=' * 30)

        train_handlers.extend([
            StatsHandler(tag_name=TRAIN_LOSS, output_transform=from_engine(['loss'], first=True)),
        ])
        
        val_handlers.extend([
            StatsHandler(
                name='evaluator',
                output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
                global_epoch_transform=lambda x: trainer.state.epoch,
            ),
            CheckpointSaver(
                save_dir=MODEL_SAVE_DIR,
                save_dict={'net': net, 'opt': opt, 'lr_scheduler': lr_scheduler},
                save_key_metric=True,
                key_metric_negative_sign=True,
                key_metric_name=VAL_LOSS,
                key_metric_filename=MODEL_SAVE_FILE_NAME,
            ),
        ])
    
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        amp=True,
        metric_cmp_fn=lambda curr, best: curr < best,
        key_val_metric={
            VAL_LOSS: Loss(loss_fn(), output_transform=from_engine(['pred', 'label']), device=device),
        },
        val_handlers=val_handlers,
    )

    test_handlers.extend([
        CheckpointLoader(
            load_path=os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_FILE_NAME),
            load_dict={'net': net, 'opt': opt, 'lr_scheduler': lr_scheduler},
        ),
    ])

    test_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        inferer=SimpleInferer(),
        amp=True,
        key_val_metric={
            TEST_LOSS: Loss(loss_fn(), output_transform=from_engine(['pred', 'label']), device=device),
        },
        additional_metrics={
            'SaveOutputs': SaveOutputs(output_transform=from_engine(['pred', 'label']), device=device),
        },
        val_handlers=test_handlers,
    )

    train_handlers.extend([
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        MetricLogger(evaluator=evaluator),
        ValidationHandler(1, evaluator),
    ])

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss_fn(),
        inferer=SimpleInferer(),
        amp=True,
        key_train_metric={
            TRAIN_LOSS: Loss(loss_fn(), output_transform=from_engine(['pred', 'label']), device=device),
        },
        train_handlers=train_handlers,
    )

    if idist.get_rank() == 0:
        print('=' * 10 + ' trainer.run() ' + '=' * 10)

    trainer.run()

    if idist.get_rank() == 0:
        print('=' * 10 + ' test_evaluator.run() ' + '=' * 10)

    test_evaluator.run()

    if idist.get_rank() == 0:
        outputs, targets = test_evaluator.state.metrics['SaveOutputs']
        performance =  get_performances(outputs, targets)        
        model_complexity_info = get_model_complexity_info(net, test_loader.dataset[0][0].shape, print_per_layer_stat=False)
        
        # Save results
        csv_logger(
            DATETIME_NOW + '_' + args.result_file, 
            ExperimantResult(
                experiment_date=DATETIME_NOW,
                test_date=TRAIN_TIME,
                iteration=iteration,
                model_name=args.model,
                loss_function=args.loss,
                loss=test_evaluator.state.metrics[TEST_LOSS],
                auroc=performance[0],
                auprc=performance[1],
                accuracy=performance[2],
                macro_f1=performance[3],
                sensitivity=performance[4],
                specificity=performance[5],
                precision=performance[6],
                params=model_complexity_info[1],
                flops=model_complexity_info[0]*2,
                best_epoch=evaluator.state.best_metric_epoch,
                batch_size=train_loader.batch_size,
            )
        )

    ddp_finalize()


def setting_reproducibility(seed):
    '''
    setting reproducibility except avg_pool3d_backward_cuda
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    '''
    This operation is not deterministic because it uses CuBLAS and you have 
    CUDA >= 10.2. To enable deterministic behavior in this case, you must set 
    an environment variable before running your PyTorch application: 
    CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. 
    For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

    RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation,
    but you set 'torch.use_deterministic_algorithms(True)'.
    You can turn off determinism just for this operation if that's acceptable for your application.
    You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding
    deterministic support for this operation.
    '''
    # torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setting_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-et', '--experiment_time', default=None, type=str, help='experiment time')
    parser.add_argument('-i', '--iteration', default=30, type=int, help='experiment iteration number')
    parser.add_argument('-d', '--data_dir', default='./Datasets/input_example', type=str, help='directory of data')
    parser.add_argument('-lf', '--label_file', default='./Datasets/target_example.csv', type=str, help='label csv file')
    #parser.add_argument('-d', '--data_dir', default='./Datasets/severity', type=str, help='directory of data')
    #parser.add_argument('-lf', '--label_file', default='./Datasets/severity.csv', type=str, help='label csv file')
    parser.add_argument('-tts', '--train_test_split_ratio', default=0.8, type=float, help='train / test split ratio')
    parser.add_argument('-tvs', '--train_val_split_ratio', default=0.75, type=float, help='train / valid split ratio')
    parser.add_argument('-m', '--model', default='DenseNet121', type=str, help='model')
    parser.add_argument('-l', '--loss', default='CrossEntropyLoss', type=str, help='loss function')
    parser.add_argument('-opt', '--optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('-sch', '--lr_scheduler', default='StepLR', type=str, help='lr scheduler')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='train & valid batch size')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-o', '--result_file', default='result.csv', type=str, help='train result file')
    parser.add_argument('-s', '--seed', default=42, type=int, help='seed')
    parser.add_argument('-ep', '--max_epochs', default=5, type=int, help='seed')
    parser.add_argument('--reproducibility', default=False, action='store_true', help='enable reproducibility')
    parser.add_argument('--model_name', default=None, type=str, help='model name')
    return parser.parse_args()


def main():
    setup_logger(level=logging.INFO)

    args = setting_arguments()

    # For Reproducibility
    if args.reproducibility:
        setting_reproducibility(args.seed)
        
    train(args)

if __name__ == '__main__':
    main()
