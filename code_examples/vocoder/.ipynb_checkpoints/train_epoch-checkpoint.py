
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
from datetime import datetime

import torch
import torch.nn as nn 

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from parser_config import config_all

from utils import get_dataset_filelists


from utils import count_parameters, count_parameters_simple,  tensor_memsize
from utils import init_weights, apply_weight_norm 
from utils import count_parameters,  tensor_memsize
from utils import copy_sourcefile 
from utils import  reduce_tensor,  init_distributed
from utils import last_checkpoint, save_checkpoint, load_checkpoint, configure_optimizer, adjust_learning_rate, adjust_loss_ratio, calculate_performance, OutputSaver
import lamb


def train_epoch(model, para_model, optimizer, scaler, criterion, args, data_config, distributed_run, device, epoch, total_iter, train_loader ) :    
    model.train()
    train_loss =0
    for i, batch in enumerate(train_loader):
        tic = time.time()
        model.zero_grad(set_to_none=True)
        total_iter += 1  
        adjust_learning_rate(total_iter, optimizer, args )   
        alpha, beta = adjust_loss_ratio(total_iter, args)
        inputs, targets = batch 
        targets = targets.to(args.local_rank).float()
        enable_autocast = args.fp16 and args.amp == 'pytorch'
        with torch.cuda.amp.autocast(enable_autocast):              
            outputs = para_model(inputs) 
            loss  = criterion(torch.squeeze(outputs,1), targets)
        if args.fp16:
            scaler.scale(loss).backward()
        else:  
            loss.backward()
        if args.fp16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.fp16 and args.amp == 'pytorch':
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()                            
        toc = time.time()
        dur = toc - tic
        if args.local_rank ==0:
            loss_val = loss.float().item()                         
            lr_print   = optimizer.param_groups[0]['lr']
    train_loss = loss.float().item()
    return model, optimizer, scaler, total_iter, train_loss  

def sample_epoch_infer(model, para_model, scaler, criterion, args,  args_data, distributed_run, device, epoch, valid_loader, output_saver) :
    model.eval()
    with torch.no_grad():     
        for i, batch in enumerate(valid_loader):
            inputs, targets = batch
            targets = targets.to(args.local_rank).float()
            enable_autocast = args.fp16 and args.amp == 'pytorch'
            with torch.cuda.amp.autocast(enable_autocast):
                tic_infer = time.time()
                if distributed_run :                        
                    outputs = para_model(inputs)
                else :
                    outputs = para_model(inputs)
                loss = np.round(criterion(torch.squeeze(outputs,1), targets).cpu().numpy(),6)
#                 auroc, auprc, acc, f1, ss, sp, pr = calculate_performance(outputs.cpu().numpy(), targets.cpu().numpy())
#                 if args.local_rank==0 :
#                     now = datetime.now()
#                     infer_date = now.strftime("%y%m%d_%H%M%S")
#                     print(str(epoch)+'EP('+infer_date+'):',end=' ')
#                     print(auroc, auprc, acc, f1, ss, sp, pr)
#                     sample_path = os.path.join(args.sample_dir)
                output_saver.update(outputs.cpu().numpy(), targets.cpu().numpy())
    return output_saver