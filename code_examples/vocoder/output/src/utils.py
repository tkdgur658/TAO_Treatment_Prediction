
import math
import os
import random
import torch
import torch.utils.data
import numpy as np

import glob
import os
import time
import matplotlib
import matplotlib.pylab as plt
from torch import nn

import lamb

import torch
import torch.nn as nn 

from torch.nn.utils import weight_norm

def tensor_memsize(target):
    import sys
    mem_size = target.nelement()   * target.element_size() 
    print(" shape{} Memory {} Byte  {:4.1f} KB {:4.2f} MB {:4.2f} GB ".format (target.shape,  mem_size , mem_size/1024, mem_size/(1024*1024), mem_size/(1024*1024*1024)   )  )
    
def count_parameters_simple(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num  params {:4.2f} M Params".format( num_params/1000000 ))

def count_parameters(model, DEBUG=True):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters",  "K Parameters",  "M Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, "{:>12d}".format(param), "{:>8.1f}".format(param/1000), "{:>3.1f}".format(param/1000000)])
        total_params+=param
    if DEBUG:
        print(table)
        print("Total Trainable Params: {:>12d} {:>8.2f}K {:>8.2f}M".format(total_params, total_params/1000, total_params/1000000) )
    return total_params    
    

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)

def to_gpu(x):
    import torch
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def copy_sourcefile(args, src_dir = 'src' ):    
    import os 
    import shutil
    import glob 
    source_dir = os.path.join(args.output_dir, src_dir)

    if not os.path.exists(source_dir):
        os.makedirs(source_dir)  
    org_files1 = os.path.join('./', '*.py' )
    org_files2 = os.path.join('./', '*.sh' )
    org_files3 = os.path.join('./', '*.ipynb' )
    org_files4 = os.path.join('./', '*.txt' )
    org_files5 = os.path.join('./', '*.json' )    
    files =[]
    files = glob.glob(org_files1 )
    files += glob.glob(org_files2  )
    files += glob.glob(org_files3  )
    files += glob.glob(org_files4  ) 
    files += glob.glob(org_files5  )     

    print("COPY source to output/source dir ", files)
    tgt_files = os.path.join( source_dir, '.' )
    for i, file in enumerate(files):
        shutil.copy(file, tgt_files)
                          
        

def get_dataset_filelists(args): 
    
    with open(args.filelists_train, 'r', encoding='utf-8') as fi:
        train_files = [os.path.join(args.dataset_dir, args.wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]
    with open(args.filelists_valid, 'r', encoding='utf-8') as fi:
        valid_files = [os.path.join(args.dataset_dir, args.wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    with open(args.filelists_test, 'r', encoding='utf-8') as fi:
        test_files = [os.path.join(args.dataset_dir, args.wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]        
    return sorted(train_files), sorted(valid_files), sorted(test_files)

def reduce_tensor(tensor, num_gpus):
    import torch
    import torch.distributed as dist    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)

def init_distributed(args, world_size, rank):
    import torch
    import torch.distributed as dist
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())
    # Initialize distributed communication
    backend = 'nccl' if args.cuda else 'gloo'
    dist.init_process_group(backend=backend,init_method='env://', rank = rank, world_size = world_size)
       
    print("Done initializing distributed training")




def last_checkpoint(output):
    import glob

    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            print(f'WARNING: Cannot load {fpath}')
            return True

    saved = sorted(
        glob.glob(f'{output}/checkpoint_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None

def save_checkpoint(model, optimizer,scaler, args, model_config, data_config, epoch, total_iter, loss):    
    import os
    import shutil    
    
    if args.local_rank != 0:
        return

    if args.fp16:
        if args.amp == 'pytorch':
            amp_state = scaler.state_dict()
        elif args.amp == 'apex':
            amp_state = amp.state_dict()
    else:
        amp_state = None
    
    state = {
        'args'           : args,
        'model_config'   : model_config,
        'data_config'    : data_config,        
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'amp_state'      : amp_state,
        'epoch'          : epoch,
        'total_iter'     : total_iter,
        'val_loss'       : loss,
        }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    filepath = os.path.join(args.output_dir, 'checkpoint_{}.pt'.format(epoch) )
    torch.save(state, filepath)   

    last_chkpt_filepath = os.path.join(args.output_dir, 'checkpoint_last.pt')
    shutil.copy(filepath, last_chkpt_filepath)    
    
def load_checkpoint(args, model, optimizer, scaler, start_epoch, start_iter):
    import os
    path = os.path.join(args.output_dir, 'checkpoint_last.pt')
    dst = f'cuda:{torch.cuda.current_device()}'
    checkpoint = torch.load(path, map_location=dst)
   
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if args.fp16:
        if args.amp == 'pytorch':
            scaler.load_state_dict(checkpoint['amp_state'])
        elif args.amp == 'apex':
            amp.load_state_dict(checkpoint['amp_state'])
    #args         = checkpoint['args']
    #model_config = checkpoint['model_config']
    #data_config  = checkpoint['data_config']
    #data_config  = checkpoint['data_config']
    val_loss     = checkpoint['val_loss']    
    start_epoch[0] = checkpoint['epoch'] + 1
    start_iter[0] = checkpoint['total_iter']
    if args.local_rank ==0:
        print("\nDEBUG : {} : load {} and resume epoch {:d} total iter : {:d} with loss {:4.8f}".format(args.local_rank, path, start_epoch[0], start_iter[0]+1, val_loss  ) )

def configure_optimizer(model, args) :
    import torch.optim as optim
    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(),     lr=args.learning_rate, momentum=args.mom)
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)        
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(),    lr=args.learning_rate,  betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'lamb':
        optimizer = lamb.Lamb(model.parameters(),     lr=args.learning_rate,  betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'jitlamb':
        optimizer = lamb.JITLamb(model.parameters(),  lr=args.learning_rate,  betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)        
    return optimizer     

def adjust_learning_rate(total_iter, opt, args):
    
    # inv_sqrt # original transformer 1000 warm up in FastPitch
    if args.warmup_steps == 0:
        scale = 1.0
    elif total_iter > args.warmup_steps:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (args.warmup_steps ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = args.learning_rate * scale
            
def adjust_loss_ratio(total_iter, args):
    alpha = 0.5          # 0.5 to 0.9  ( 0--2000 step)
    beta  = 1.0 - alpha  # 1-alpha    ( 0--2000 step)
    
    if total_iter < (args.warmup_steps*2 ):  
        scale = 1.0 + 0.8*(total_iter / (args.warmup_steps*2) )
        alpha = 0.5 *  scale
        beta = 1.0 - alpha            
    else :
        alpha = 0.9
        beta = 1.0 - alpha
    
    return alpha, beta
