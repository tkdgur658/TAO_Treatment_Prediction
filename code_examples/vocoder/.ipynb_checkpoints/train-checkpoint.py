import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

import os
import time 
import glob
import shutil
import argparse
import json
import numpy as np 
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.transforms import Compose, RandRotate
from sklearn.model_selection import train_test_split
from apex import amp

from parser_config import config_all
from utils import get_dataset_filelists
from utils import count_parameters, count_parameters_simple,  tensor_memsize
from utils import init_weights, apply_weight_norm 
from utils import count_parameters,  tensor_memsize
from utils import copy_sourcefile 
from utils import  reduce_tensor,  init_distributed
from utils import last_checkpoint, save_checkpoint, load_checkpoint, configure_optimizer, adjust_learning_rate, adjust_loss_ratio
from utils import CustomDataset, OutputSaver, calculate_performance, find_free_port, control_random_seed, str_to_class
from train_epoch import train_epoch, sample_epoch_infer
import lamb

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def main(args):
    if args.local_rank==0:
        print("args : ", args)
    distributed_run = True
    control_random_seed(args.seed)                 
    if args.local_rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    device = torch.device(args.device)
    
    # Initialize device and distributed backend
    if distributed_run:
        print("Initializing distributed training")
        init_distributed(args, args.world_size, args.local_rank)
                 
    criterion = nn.BCEWithLogitsLoss()
    exec(f'from {args.model_dir}.{args.module_name} import *')
    model = str_to_class(args.model_name)(args.in_channels, args.num_classes)
    
    if args.local_rank==0:       
        count_parameters_simple(model)
           
    optimizer = configure_optimizer(model, args)               
    model.to(device)

    ### configure AMP    
    scaler = None    
    if args.fp16:
        if args.amp == 'pytorch':
            scaler = torch.cuda.amp.GradScaler()
        elif args.amp == 'apex':
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', )
            scaler = None

    if args.multi_gpu == 'ddp':
        para_model = DDP(model, device_ids=[args.local_rank],output_device=args.local_rank,
                                broadcast_buffers=False, find_unused_parameters=True, )
        distributed_run = True
    else:
        para_model = model
        
    start_epoch = [1]
    start_iter  = [0]
            
    if args.resume:
        load_checkpoint(args, model, optimizer, scaler, start_epoch, start_iter)

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]    
    
    df_label = pd.read_csv(args.label_file, header=None, index_col=0)

    images = [os.path.join(args.data_dir, f'{i}.npy') for i in df_label.index.to_list()]
    labels = df_label.to_numpy(dtype=np.int64).flatten()
   
    train_test_split_ratio = 0.8
    train_val_split_ratio = 0.75
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, train_size=train_test_split_ratio, random_state=args.seed,
        stratify=labels if len(labels) * (1 - train_test_split_ratio) > 1 else None,
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, train_size=train_val_split_ratio, random_state=args.seed,
        stratify=train_labels if len(train_labels) * (1 - train_val_split_ratio) > 1 else None,
    )

    # Define transforms
    train_transforms = Compose([RandRotate(range_x=10, range_y=10, range_z=10)])
    val_transforms = None
    test_transforms = None

    # create a training data loader
    trainset = CustomDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    validset = CustomDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
    testset = CustomDataset(image_files=test_images, labels=test_labels, transform=test_transforms)

    if  args.multi_gpu == 'ddp' and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(trainset)
        valid_sampler = DistributedSampler(validset)        
        test_sampler  = DistributedSampler(testset)           
        suffle=False
    else :
        train_sampler = None
        valid_sampler = None
        test_sampler  = None
        suffle=False
    
    train_loader = DataLoader(trainset, num_workers=0, shuffle=False,
                          sampler=train_sampler, batch_size=args.Batch_Size ,
                          pin_memory=False, drop_last=False,
                          collate_fn=None)
    
    valid_loader = DataLoader(validset, num_workers=0, shuffle=False,
                          sampler=valid_sampler, batch_size=args.Batch_Size ,
                          pin_memory=False, drop_last=False,
                          collate_fn=None)
    
    test_loader = DataLoader(testset, num_workers=0, shuffle=False,
                          sampler=test_sampler, batch_size=args.Batch_Size ,
                          pin_memory=False, drop_last=False,
                          collate_fn=None)

    model.train()
    model.zero_grad()
    
    output_saver = OutputSaver()
    Best_Loss = 9999
    for epoch in range(start_epoch, args.Epochs+1):
        model, optimizer, scaler,  total_iter, loss =  train_epoch(
            model, para_model, optimizer, scaler, criterion, args, distributed_run, device, epoch, total_iter, train_loader)   
        
        if args.local_rank==0:
            output_saver.reset()
        output_saver = sample_epoch_infer(model, para_model, scaler, criterion, args,  distributed_run, device, epoch, valid_loader, output_saver)
        outputs, targets = output_saver.return_array()
        loss = np.round(criterion(torch.tensor(outputs), torch.tensor(targets)).cpu().numpy(),6)
        auroc, auprc, acc, f1, ss, sp, pr = calculate_performance(outputs, targets)
        now = datetime.now()
        infer_date = now.strftime("%y%m%d_%H%M%S")
        print(f'{epoch} EP({infer_date}): Loss: {loss}, AUROC: {auroc}, AUPRC: {auprc}, ACC: {acc}, F1: {f1}, SS: {ss}, SP: {sp}, PR:{pr}')
        
        if (Best_Loss>=loss and args.local_rank == 0):
            save_checkpoint(model, optimizer, scaler, args, epoch, total_iter, loss)
            if args.local_rank ==0:
                print(f"Best Epoch: {epoch}, Loss: {loss}", end='')
    if args.local_rank==0:
        print("Train End")
        


if __name__ == '__main__':

#     os.environ["MASTER_ADDR"] = "127.0.0.1"
#     port = find_free_port()
#     os.environ["MASTER_PORT"] = port
#     print('PORT:', port)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn') # solution for RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    
    parser = argparse.ArgumentParser(description='PyTorch MILK ', allow_abbrev=False)
    parser = config_all(parser)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    
    if args.local_rank ==0 :
        copy_sourcefile(args, src_dir='src')
    main(args)