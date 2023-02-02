
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

import os
import time 
import glob
import shutil
import argparse
import json

import torch
import torch.nn as nn 

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from parser_config import config_all

from torch_stft import STFT, STDCT, STRFT
from audioutil_librosa import save_wav_librosa
from melgen_torch import MILKDataset_torch
from utils import get_dataset_filelists


from utils import count_parameters, count_parameters_simple,  tensor_memsize
from utils import init_weights, apply_weight_norm 
from utils import count_parameters,  tensor_memsize
from utils import copy_sourcefile 
from utils import  reduce_tensor,  init_distributed
from utils import last_checkpoint, save_checkpoint, load_checkpoint, configure_optimizer, adjust_learning_rate, adjust_loss_ratio
import lamb


def train_epoch(model, para_model, optimizer, scaler, criterion, args, data_config, distributed_run, device, epoch, total_iter, train_loader ) :    
    from audioutil_torch import get_drc_torch, get_drc_inv_torch
    model.train()
    train_loss =0
    for i, batch in enumerate(train_loader):
        tic = time.time()
        model.zero_grad(set_to_none=True)
        total_iter += 1 
        adjust_learning_rate(total_iter, optimizer, args )   
        alpha, beta = adjust_loss_ratio(total_iter, args)
        nframes, audio_gt, mel_gt, lossy_full, rabs_gt, rsign_gt = batch 
        mel_gt     =       mel_gt.transpose(1, 2)        
        lossy_full =   lossy_full.transpose(1, 2)
        rabs_gt    =      rabs_gt.transpose(1, 2)
        rsign_gt   =     rsign_gt.transpose(1, 2)
        
        inputs = lossy_full  
        
        enable_autocast = args.fp16 and args.amp == 'pytorch'
        with torch.cuda.amp.autocast(enable_autocast):              
            rabs_pr    = para_model(inputs, DEBUG=False)        
            loss  = criterion(rabs_pr, rabs_gt)             

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
        #if args.local_rank ==0  and i% 10 ==0 :
        if args.local_rank ==0:
            loss_val = loss.float().item()                         
            lr_print   = optimizer.param_groups[0]['lr']
            print("\nep:{:4d}/{:4d}it:{:3d}/{:3d}|{:8d}|loss:MSE:{:4.6f}|lr:{:4.6f}|{:4.2f}s/it ".format(
                epoch, args.epochs, i, len(train_loader) , total_iter,  loss_val, lr_print ,  dur ), end='' )               

    train_loss = loss.float().item()
    return model, optimizer, scaler, total_iter, train_loss  

def sample_epoch_infer(model, para_model, scaler, args,  args_data, distributed_run, device, epoch, valid_loader ) :  
    import os 
    import torch 
    import matplotlib.pyplot as plt
    import IPython.display as ipd
    import numpy as np    
    import librosa 
    from audioutil_torch import get_drc_torch, get_drc_inv_torch        
    from audioutil_librosa import save_wav_librosa    
    from scipy.io.wavfile import write as write_wave

    np.set_printoptions(threshold=40)
    model.eval()
    stdct =  STDCT(filter_length=args_data.n_fft, hop_length=args_data.hop_length, window_type=args_data.win_type, pad=True, args=args, DEBUG=args.DEBUG ).to(torch.device(args.device, args.local_rank))  
 
    with torch.no_grad():     
        for i, batch in enumerate(valid_loader):
            tic = time.time()
            nframes, audio_gt, mel_gt, lossy_full, rabs_gt, rsign_gt = batch 
            mel_gt     =     mel_gt.transpose(1, 2)
            lossy_full = lossy_full.transpose(1, 2)
            rabs_gt    =    rabs_gt.transpose(1, 2)
            rsign_gt   =   rsign_gt.transpose(1, 2)
            
            inputs = lossy_full 

            enable_autocast = args.fp16 and args.amp == 'pytorch'
            with torch.cuda.amp.autocast(enable_autocast):
 
                tic_infer = time.time()
                if distributed_run :                        
                    rabs_pr = para_model.module.infer(inputs, DEBUG=False)                
                else :
                    rabs_pr = para_model.model.infer(inputs, DEBUG=False)    
                torch.cuda.synchronize()
                toc_infer = time.time()
                dur_infer = toc_infer - tic_infer

                mel_gt     =     mel_gt.transpose(1,2)                   
                lossy_full = lossy_full.transpose(1,2)                      
                rabs_gt    =    rabs_gt.transpose(1, 2)                    
                rsign_gt   =   rsign_gt.transpose(1, 2)
                tic_trans = time.time()    
                rabs_pr    =  rabs_pr.transpose(1, 2)
                #rsign_pr   = rsign_pr.transpose(1, 2)  
                torch.cuda.synchronize()
                toc_trans  = time.time()
                dur_trans = toc_trans - tic_trans                                       

                audio_gt = audio_gt.squeeze(1)
                tic_stdct = time.time()
                wav_gt_gt = stdct.inverse(rabs_gt, torch.sign(rsign_gt) ).squeeze(1)
                
                torch.cuda.synchronize()
                toc_stdct = time.time()
                dur_stdct = toc_stdct - tic_stdct 
                dur_total = dur_infer +  dur_trans +  dur_stdct

                #wav_pr_pr = stdct.inverse(rabs_pr, torch.sign(rsign_pr) ).squeeze(1)
                wav_pr_gt = stdct.inverse(rabs_pr, torch.sign(rsign_gt) ).squeeze(1)            
                #wav_gt_pr = stdct.inverse(rabs_gt, torch.sign(rsign_pr) ).squeeze(1)
                 
                if args.local_rank ==0 :
                    print("inf{:>03.1f}ms+tr{:>2.1f}ms+si{:>2.1f}ms={:>03.2f}ms {:d}fr".format(
                        dur_infer*1000, dur_trans*1000, dur_stdct*1000,  dur_total*1000, 
                        args_data.max_nframes  ), end='')
                if args.local_rank==0 :
                    sample_path = os.path.join(args.sample_dir)
                    if not os.path.exists(sample_path):
                        os.makedirs(sample_path)
                    
                    mels_filename      = os.path.join( sample_path, 'mel_{:04d}.pt'.format(epoch) )
                    wav_filename_gt    = os.path.join( sample_path, 'wav_{:04d}_gt_or.wav'.format(epoch) )                       
                    wav_filename_gt_gt = os.path.join( sample_path, 'wav_{:04d}_gt_gt.wav'.format(epoch) )
                    #wav_filename_pr_pr = os.path.join( sample_path, 'wav_{:04d}_pr_pr.wav'.format(epoch) )
                    #wav_filename_gt_pr = os.path.join( sample_path, 'wav_{:04d}_gt_pr.wav'.format(epoch) )
                    wav_filename_pr_gt = os.path.join( sample_path, 'wav_{:04d}_pr_gt.wav'.format(epoch) )    
                    
                    output_mels = (audio_gt.squeeze(1), mel_gt, lossy_full,  rabs_gt, rsign_gt, rabs_pr )
                    wav_gt    =  audio_gt.T.cpu().numpy().astype(np.float32)
                    wav_gt_gt = wav_gt_gt.T.cpu().numpy().astype(np.float32)
                    #wav_pr_pr = wav_pr_pr.T.cpu().numpy().astype(np.float32)
                    #wav_gt_pr = wav_gt_pr.T.cpu().numpy().astype(np.float32)
                    wav_pr_gt = wav_pr_gt.T.cpu().numpy().astype(np.float32) 
                    
                    torch.save(output_mels, mels_filename )
                    save_wav_librosa(wav_filename_gt,    wav_gt )
                    save_wav_librosa(wav_filename_gt_gt, wav_gt_gt )
                    #save_wav_librosa(wav_filename_pr_pr, wav_pr_pr )
                    #save_wav_librosa(wav_filename_gt_pr, wav_gt_pr )
                    save_wav_librosa(wav_filename_pr_gt, wav_pr_gt )

                toc = time.time()
                dur = toc - tic 
                if args.DEBUG and  args.local_rank==0 : 
                    print("DEBUG : sample_epoch_iter 12 infer try {} {:4.2f}ms ".format( i, dur*1000  )  , end='')
            

def train_epoch_nse(model, para_model, optimizer, scaler, criterion, args, data_config, distributed_run, device, epoch, total_iter, train_loader ) :    
    from audioutil_torch import get_drc_torch, get_drc_inv_torch
    model.train()
    train_loss =0
    for i, batch in enumerate(train_loader):
        tic = time.time()
        model.zero_grad(set_to_none=True)
        total_iter += 1 
        adjust_learning_rate(total_iter, optimizer, args )   
        alpha, beta = adjust_loss_ratio(total_iter, args)
        nframes, audio_gt, mel_gt, lossy_full, rabs_gt, rsign_gt = batch 
        mel_gt     =       mel_gt.transpose(1, 2)        
        lossy_full =   lossy_full.transpose(1, 2)
        rabs_gt    =      rabs_gt.transpose(1, 2)
        rsign_gt   =     rsign_gt.transpose(1, 2)
        
        inputs = rabs_gt  
        
        enable_autocast = args.fp16 and args.amp == 'pytorch'
        with torch.cuda.amp.autocast(enable_autocast):              
            rsign_pr    = para_model(inputs, DEBUG=False)        
            loss  = criterion(torch.sign(rsign_pr), torch.sign(rsign_gt) )             

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
        if args.local_rank ==0  and i% 10 ==0 :
            loss_val = loss.float().item()                         
            lr_print   = optimizer.param_groups[0]['lr']
            print("\nep:{:4d}/{:4d}it:{:3d}/{:3d}|{:8d}|loss:MSE:{:4.6f}|lr:{:4.6f}|{:4.2f}s/it ".format(
                epoch, args.epochs, i, len(train_loader) , total_iter,  loss_val, lr_print ,  dur ), end='' )               

    train_loss = loss.float().item()
    return model, optimizer, scaler, total_iter, train_loss  

def sample_epoch_infer_nse(model, para_model, scaler, args,  args_data, distributed_run, device, epoch, valid_loader ) :  
    import os 
    import torch 
    import matplotlib.pyplot as plt
    import IPython.display as ipd
    import numpy as np    
    import librosa 
    from audioutil_torch import get_drc_torch, get_drc_inv_torch        
    from audioutil_librosa import save_wav_librosa    
    from scipy.io.wavfile import write as write_wave

    np.set_printoptions(threshold=40)
    model.eval()
    stdct =  STDCT(filter_length=args_data.n_fft, hop_length=args_data.hop_length, window_type=args_data.win_type, pad=True, args=args, DEBUG=args.DEBUG ).to(torch.device(args.device, args.local_rank))  
 
    with torch.no_grad():     
        for i, batch in enumerate(valid_loader):
            tic = time.time()
            nframes, audio_gt, mel_gt, lossy_full, rabs_gt, rsign_gt = batch 
            mel_gt     =     mel_gt.transpose(1, 2)
            lossy_full = lossy_full.transpose(1, 2)
            rabs_gt    =    rabs_gt.transpose(1, 2)
            rsign_gt   =   rsign_gt.transpose(1, 2)
            
            inputs = rabs_gt 

            enable_autocast = args.fp16 and args.amp == 'pytorch'
            with torch.cuda.amp.autocast(enable_autocast):
 
                tic_infer = time.time()
                if distributed_run :                        
                    rsign_pr = para_model.module.infer(inputs, DEBUG=False)                
                else :
                    rsign_pr = para_model.model.infer(inputs, DEBUG=False)    
                torch.cuda.synchronize()             
                toc_infer = time.time()
                dur_infer = toc_infer - tic_infer

                mel_gt     =     mel_gt.transpose(1,2)                   
                lossy_full = lossy_full.transpose(1,2)                      
                rabs_gt    =    rabs_gt.transpose(1, 2)                    
                rsign_gt   =   rsign_gt.transpose(1, 2)
                tic_trans = time.time()    
                #rabs_pr    =  rabs_pr.transpose(1, 2)
                rsign_pr   = rsign_pr.transpose(1, 2)  
                torch.cuda.synchronize()
                toc_trans  = time.time()
                dur_trans = toc_trans - tic_trans                                       

                audio_gt = audio_gt.squeeze(1)
                tic_stdct = time.time()
                wav_gt_gt = stdct.inverse(rabs_gt, torch.sign(rsign_gt) ).squeeze(1)
                
                torch.cuda.synchronize()
                toc_stdct = time.time()
                dur_stdct = toc_stdct - tic_stdct 
                dur_total = dur_infer +  dur_trans +  dur_stdct

                #wav_pr_pr = stdct.inverse(rabs_pr, torch.sign(rsign_pr) ).squeeze(1)
                #wav_pr_gt = stdct.inverse(rabs_pr, torch.sign(rsign_gt) ).squeeze(1)            
                wav_gt_pr = stdct.inverse(rabs_gt, torch.sign(rsign_pr) ).squeeze(1)
                 
                if args.local_rank ==0 :
                    print("inf{:>03.1f}ms+tr{:>2.1f}ms+si{:>2.1f}ms={:>03.2f}ms {:d}fr".format(
                        dur_infer*1000, dur_trans*1000, dur_stdct*1000,  dur_total*1000, 
                        args_data.max_nframes  ), end='')
                if args.local_rank==0 :
                    sample_path = os.path.join(args.sample_dir)
                    if not os.path.exists(sample_path):
                        os.makedirs(sample_path)
                    
                    mels_filename      = os.path.join( sample_path, 'mel_{:04d}.pt'.format(epoch) )
                    wav_filename_gt    = os.path.join( sample_path, 'wav_{:04d}_gt_or.wav'.format(epoch) )                       
                    wav_filename_gt_gt = os.path.join( sample_path, 'wav_{:04d}_gt_gt.wav'.format(epoch) )
                    #wav_filename_pr_pr = os.path.join( sample_path, 'wav_{:04d}_pr_pr.wav'.format(epoch) )
                    wav_filename_gt_pr = os.path.join( sample_path, 'wav_{:04d}_gt_pr.wav'.format(epoch) )
                    #wav_filename_pr_gt = os.path.join( sample_path, 'wav_{:04d}_pr_gt.wav'.format(epoch) )    
                    
                    output_mels = (audio_gt.squeeze(1), mel_gt, lossy_full,  rabs_gt, rsign_gt, rsign_pr )
                    wav_gt    =  audio_gt.T.cpu().numpy().astype(np.float32)
                    wav_gt_gt = wav_gt_gt.T.cpu().numpy().astype(np.float32)
                    #wav_pr_pr = wav_pr_pr.T.cpu().numpy().astype(np.float32)
                    wav_gt_pr = wav_gt_pr.T.cpu().numpy().astype(np.float32)
                    #wav_pr_gt = wav_pr_gt.T.cpu().numpy().astype(np.float32) 
                    
                    torch.save(output_mels, mels_filename )
                    save_wav_librosa(wav_filename_gt,    wav_gt )
                    save_wav_librosa(wav_filename_gt_gt, wav_gt_gt )
                    #save_wav_librosa(wav_filename_pr_pr, wav_pr_pr )
                    save_wav_librosa(wav_filename_gt_pr, wav_gt_pr )
                    #save_wav_librosa(wav_filename_pr_gt, wav_pr_gt )

                toc = time.time()
                dur = toc - tic 
                if args.DEBUG and  args.local_rank==0 : 
                    print("DEBUG : sample_epoch_iter 12 infer try {} {:4.2f}ms ".format( i, dur*1000  )  , end='')
            
