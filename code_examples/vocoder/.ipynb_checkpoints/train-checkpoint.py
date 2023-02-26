
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

import torch
import torch.nn as nn 

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.model_selection import train_test_split

from apex import amp


from parser_config import config_all
from utils import get_dataset_filelists

#from model import MILK_FFMixer_2stage
from Fake_Model import Fake_Model
from ResNet import ResNet18
from ResNet_3D import ResNet18_3D
from utils import count_parameters, count_parameters_simple,  tensor_memsize
from utils import init_weights, apply_weight_norm 
from utils import count_parameters,  tensor_memsize
from utils import copy_sourcefile 
from utils import  reduce_tensor,  init_distributed

from utils import last_checkpoint, save_checkpoint, load_checkpoint, configure_optimizer, adjust_learning_rate, adjust_loss_ratio
from utils import CustomDataset
import lamb

from train_epoch import train_epoch, sample_epoch_infer

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def main(args, args_data, args_model):
    print("DEBUG : start main")  
    
    if args.local_rank==0:
        print("args : ", args)
        print("args_data :  ", args_data)
        print("args_model : ", args_model)
    #distributed_run = args.world_size > 1
    distributed_run = True
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)    

    if args.local_rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    device = torch.device(args.device)
    
    # Initialize device and distributed backend
    if distributed_run:
        print("Initializing distributed training")
        init_distributed(args, args.world_size, args.local_rank)
        
    if args.local_rank==0:
        print("DEBUG : configure loss")             
    criterion = nn.BCEWithLogitsLoss()
    
    if args.local_rank==0:        
        print("DEBUG : configure model")  
    model = ResNet18_3D(1, 1)
    
    if args.local_rank==0:       
        count_parameters_simple(model)
           
    optimizer = configure_optimizer(model, args)               
    model.to(device)

    ### configure AMP    
    if args.local_rank==0:
        print("DEBUG : AMP config" )
    scaler = None    
    if args.fp16:
        if args.amp == 'pytorch':
            scaler = torch.cuda.amp.GradScaler()
        elif args.amp == 'apex':
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', )
            scaler = None

    ### configure distribute
    print('model to DDP model')
    if args.local_rank==0:
        print("DEBUG : DDP config" )    
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
    
    
    ### data load
    if args.local_rank==0:
        print("DEBUG : data loader" )      
    
    ##############################################################################################################################
    data_dir = 'input_example'
    label_file = 'target_example.csv'
    
    df_label = pd.read_csv(label_file, header=None, index_col=0)

    images = [os.path.join(data_dir, f'{i}.npy') for i in df_label.index.to_list()]
    labels = df_label.to_numpy(dtype=np.int64).flatten()

   # Train/Valid/Test split
    seed = 0
    from monai.transforms import Compose, RandRotate
    
    train_test_split_ratio = 0.8
    train_val_split_ratio = 0.75
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, train_size=train_test_split_ratio, random_state=seed,
        stratify=labels if len(labels) * (1 - train_test_split_ratio) > 1 else None,
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, train_size=train_val_split_ratio, random_state=seed,
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
    
    ##############################################################################################################################
    
    print('DataLoader Generation')
    
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
                          sampler=train_sampler, batch_size=args.batch_size ,
                          pin_memory=False, drop_last=False,
                          collate_fn=None )   
    
    valid_loader = DataLoader(validset, num_workers=0, shuffle=False,
                          sampler=valid_sampler, batch_size=args.batch_size ,
                          pin_memory=False, drop_last=False,
                          collate_fn=None )   
    
    test_loader = DataLoader(testset, num_workers=0, shuffle=False,
                          sampler=test_sampler, batch_size=args.batch_size ,
                          pin_memory=False, drop_last=False,
                          collate_fn=None )        

    if args.local_rank ==0:
        print("DEBUG : train_loader : ",  len(train_loader) )
        print("DEBUG : valid_loader : ",  len(valid_loader) )
        print("DEBUG : test_loader : ",  len(test_loader) )        
        
    model.train()   
    model.zero_grad()
    
    if args.local_rank ==0:
        print("DEBUG : start epoch===================================")  
    for epoch in range(start_epoch, args.epochs+1):
        tic_epoch = time.time()

        model, optimizer, scaler,  total_iter, loss =  train_epoch(
            model, para_model, optimizer, scaler, criterion, args, args_data, distributed_run, device, epoch, total_iter, train_loader  )   
        if args.local_rank ==0 and args.DEBUG : 
            print("DEBUG : train_epoch done")
            
        toc_epoch = time.time()
        dur_epoch = toc_epoch - tic_epoch               
        
        if args.local_rank ==0 : 
            print(" | {:4.2f}sec/epoch loss : {:4.8f} ".format(dur_epoch, loss), end='') 

        if (epoch > 0 and args.epochs_per_checkpoint > 0 and  (epoch % args.epochs_per_checkpoint == 0) and args.local_rank == 0):
            save_checkpoint(model, optimizer,scaler, args, args_model, args_data, epoch, total_iter, loss)
            if args.local_rank ==0: 
                print(" checkpoint saved", end='')
            
        if (epoch > 0 and args.epochs_per_checkpoint > 0) and   (epoch % args.epochs_per_checkpoint == 0) :            
            sample_epoch_infer( model, para_model, scaler, args,  args_data, distributed_run, device, epoch, valid_loader   )  
            if args.local_rank==0:
                print(" sample saved", end='')
    
        model.train() 
    if args.local_rank==0:
        print("DEBUG : train finished")
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

if __name__ == '__main__':

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    port = find_free_port()
    os.environ["MASTER_PORT"] = port
    print('PORT:', port)
    
    torch.backends.cudnn.enabled = True
    
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn') # solution for RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    
    parser = argparse.ArgumentParser(description='PyTorch MILK ', allow_abbrev=False)    
    parser.add_argument(      '--task',   type=str, default='train', choices=['dataset', 'train', 'infer', 'infer_e2e' ],  help='')           
    parser.add_argument('-c', '--config', type=str, default='config.json', help='JSON file for configuration')     
    parser.add_argument('--DEBUG',         action='store_true',       help='DEBUG mode default is False')         
    parser.add_argument(      '--device',  type=str, default='cuda',                   help='force CPU mode for debug')    
       
    parser = config_all(parser)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    
    
    if args.local_rank==0 and args.DEBUG:
        print("DEBUG : PyTorch MILK ")
 
    with open(args.config) as f:
        data = f.read()    

    config_json = json.loads(data)
    config_data   = config_json["data_config"]
    config_model  = config_json["model_config"]   
    
    args_data  =  AttrDict(config_data)
    args_model =  AttrDict(config_model)
    if args.local_rank ==0 :
        copy_sourcefile(args, src_dir='src')
    
    main(args, args_data, args_model)