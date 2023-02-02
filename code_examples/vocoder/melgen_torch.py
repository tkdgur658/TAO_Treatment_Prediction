
import os
import math
import random
import torch
import torch.utils.data
import time
import numpy as np
import librosa

from audioutil_librosa import load_wav_librosa, adjust_wav_librosa, norm_wav_librosa, get_wav_fromfile_librosa, enpad_wav_librosa, depad_wav_librosa

from audioutil_torch import load_wav_torch, adjust_wav_torch, norm_wav_torch, save_wav_torch, resample_wav_torch, enpad_wav_torch, depad_wav_torch
from audioutil_torch import get_magphase_torch, get_magphase_inv_torch, get_magangle_torch, get_magangle_inv_torch 
from audioutil_torch import get_window_torch,get_lin2mel_full_torch, get_lin2mel_half_torch, get_mel2lin_half_torch,get_mel2lin_full_torch,   get_mel_basis_half_torch, get_mel_basis_full_torch
from audioutil_torch import get_abssign_torch, get_abssign_inv_torch, get_zeroclip_torch
from audioutil_torch import get_drc_torch, get_drc_inv_torch

from torch_stft import STFT, STDCT, STRFT

from utils import get_dataset_filelists

MAX_WAV_VALUE = 32768.0       
    
class MILKDataset_torch(torch.utils.data.Dataset):
    def __init__(self, filelists, args, data_config ):    
        self.audio_files = filelists        
        random.seed(1234)
        if args.shuffle:
            random.shuffle(self.audio_files)            
            
        self.device            = torch.device(args.device, args.local_rank)
        self.loadfromfile      = args.loadfromfile
        self.mel_gen_option    = args.meltofile        
        self.base_wavs_path    = args.wavs_dir        
        self.base_mels_path    = args.mels_dir        
        self.ext_lin_mag       = args.ext_mel_file        
        self.spec_option       = args.spec_option 
        self.n_cache_reuse     = args.n_cache_reuse        
        
        self.sr                = data_config.sr
        self.split             = data_config.split           
        self.segment_size      = data_config.max_nframes * data_config.hop_length 
        
        self.n_fft             = data_config.n_fft
        self.hop_length        = data_config.hop_length   #  use hop_length in other code
        self.win_length        = None
        self.win_type          = data_config.win_type        
        self.n_mels            = data_config.n_mels        
        self.fmin              = data_config.fmin
        self.fmax              = data_config.fmax
        self.amin              = data_config.amin
        self.amax              = data_config.amax                   
 
        self.norm_mel          = data_config.norm_mel
        self.norm_lossy        = data_config.norm_lossy        
        self.norm_mag          = data_config.norm_mag
        self.norm_rabs         = data_config.norm_rabs
        self.norm_rzc          = data_config.norm_rzc  

        self.pad_nums          = data_config.pad_nums                       

        self.cached_wav        = None        
        self._cache_ref_count  = 0

        self.DEBUG             = args.DEBUG

        self.window_val        = get_window_torch(win_length=self.n_fft, win_type=self.win_type).to(torch.device(self.device))
        self.mel_basis_half    = get_mel_basis_half_torch( sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin , fmax=self.fmax, DEBUG=self.DEBUG ).to(self.device)
        self.mel_basis_half_T  = get_mel_basis_half_torch( sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin , fmax=self.fmax, DEBUG=self.DEBUG ).T.to(self.device)        
        self.mel_basis_full    = get_mel_basis_full_torch( sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin , fmax=self.fmax, DEBUG=self.DEBUG ).to(self.device)    
        self.mel_basis_full_T  = get_mel_basis_full_torch( sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin , fmax=self.fmax, DEBUG=self.DEBUG ).T.to(self.device) 
        
        self.stft  =   STFT(filter_length=self.n_fft, hop_length=self.hop_length, window_type=self.win_type, pad=True, args=args, DEBUG=self.DEBUG ).to(self.device) 
        self.stdct =  STDCT(filter_length=self.n_fft, hop_length=self.hop_length, window_type=self.win_type, pad=True, args=args, DEBUG=self.DEBUG ).to(self.device) 
        self.strft =  STRFT(filter_length=self.n_fft, hop_length=self.hop_length, window_type=self.win_type, pad=True, args=args, DEBUG=self.DEBUG ).to(self.device) 
        
        if self.DEBUG :
            print("DEBUG : init data loader")
            print("self.window_val       ", self.window_val.shape  )
            print("self.mel_basis_half   ", self.mel_basis_half.shape   )
            print("self.mel_basis_half_T ", self.mel_basis_half_T.shape   )
            print("self.mel_basis_full   ", self.mel_basis_full.shape   )
            print("self.mel_basis_full_T ", self.mel_basis_full_T.shape   )            
         
  
    
    def __getitem__(self, index):
        
     
        filepath = self.audio_files[index]          
        if self.DEBUG :
            print ("DEBUG : idx, file" ,index, filepath )

        audio = get_wav_fromfile_librosa(filepath, sr=self.sr, hop_length=self.hop_length,  pad_nums=self.pad_nums, mono=True, dur=15) 
        if self.DEBUG :
            print ("DEBUG : len audio ", len(audio) )
            import IPython.display as ipd
            ipd.display(ipd.Audio(audio, rate=self.sr))                                   
              
        if self.split:
            if self.DEBUG :
                print("DEBUG start split")
                
            if len(audio) >= self.segment_size:
                if self.DEBUG :
                    print ("DEBUG :audio is large, trim")
                    
                max_audio_start = len(audio) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                if self.DEBUG :
                    print(audio.shape)
                audio = audio[ audio_start:audio_start+self.segment_size]
                
                if self.DEBUG :
                    print ("DEBUG : segment  ", len(audio),self.segment_size, max_audio_start, audio_start,   )                
                    import IPython.display as ipd
                    ipd.display(ipd.Audio(audio, rate=self.sr))
            else:
                if self.DEBUG :
                    print ("DEBUG :audio is small, pad")                
                audio = librosa.util.fix_length(audio, self.segment_size)
                if self.DEBUG :
                    print ("DEBUG : segment  ", len(audio),self.segment_size    )                
                    import IPython.display as ipd
                    ipd.display(ipd.Audio(audio, rate=self.sr))                
        nsamples = len(audio)
        #audio = librosa.util.pad_center(audio, len(audio) + self.pad_nums * 2 * self.hop_length  , mode='constant')
        
        #audio = librosa.util.fix_length(audio, self.segment_size + self.pad_nums * self.hop_length)
        nsamples_pad = len(audio)
        audio = torch.FloatTensor(audio).unsqueeze(0).to(torch.device(self.device))     
                    

        mag_gt, ang_gt    = self.stft.transform(audio) 
        nframes = mag_gt.shape[2] # batch x nbins x nframes
        
        if self.DEBUG :
            print ("DEBUG :  mag_gt.shape  ang_gt.shape  ", mag_gt.shape, ang_gt.shape  )               
        rabs_gt, rsign_gt = self.stdct.transform(audio)   
        if self.DEBUG :
            print ("DEBUG :  rabs_gt.shape  rsign_gt.shape  ", rabs_gt.shape, rsign_gt.shape  )                
        
        if self.DEBUG :
            print("DEBUG 1  dataloader :  mag_gt.shape, mel_basis_half.shape ", mag_gt.shape, self.mel_basis_half.shape)
            

        mel_gt = get_lin2mel_half_torch(mag_gt, mel_basis = self.mel_basis_half, sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, DEBUG=self.DEBUG)         
        if self.norm_mel : 
            mel_gt = get_drc_torch(mel_gt)      
        mel_gt_out = mel_gt    
        if self.DEBUG :
            print("DEBUG 2  dataloader :  after lin2mel mel_gt, mel_gt_out", mel_gt.shape, mel_gt_out.shape)
              
        
        if self.norm_mag : 
            mag_gt = get_drc_torch(mag_gt)     
        mag_gt_out = mag_gt         
        
        if self.DEBUG :
            print("DEBUG 3 dataloader :  after drc mag_gt ", mag_gt.shape, mag_gt_out.shape)
              
        
        
        if self.norm_mel : 
            mel_gt = get_drc_inv_torch(mel_gt)      
        
        mel_gt_in = mel_gt  
        if self.DEBUG :
            print("DEBUG 4 dataloader :  after drd mag_gt ",   mel_gt.shape)
             
                
        mag_lossy_half = get_mel2lin_half_torch(mel_gt_in, mel_basis_T = self.mel_basis_half_T, sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, DEBUG=self.DEBUG)  # 513       
        if self.DEBUG :
            print("DEBUG 5 dataloader :  after get_mel2lin_half_torch mag_gt ",  self.mel_basis_half_T.shape, mel_gt.shape ,  mag_lossy_half.shape)
            
        mag_lossy_full = get_mel2lin_full_torch(mel_gt_in, mel_basis_T = self.mel_basis_full_T, sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, DEBUG=self.DEBUG)  #1024            
        if self.DEBUG :
            print("DEBUG 6  dataloader :  after get_mel2lin_half_torch mag_gt ",  self.mel_basis_full_T.shape, mel_gt.shape ,  mag_lossy_full.shape)
            
        if self.norm_lossy : 
            mag_lossy_half = get_drc_torch(mag_lossy_half)  
        mag_lossy_half_out = mag_lossy_half            
            
        if self.DEBUG :
            print("DEBUG 7  dataloader : after drc  mag_lossy_half.shape ",  mag_lossy_half_out.shape)        
        
        
        if self.norm_lossy :  
            mag_lossy_full = get_drc_torch(mag_lossy_full)             
        mag_lossy_full_out = mag_lossy_full   
        if self.DEBUG :
            print("DEBUG 8 dataloader :   after drc mag_lossy_full.shape ", mag_lossy_full_out.shape)        
                
      
        rzc_gt = get_zeroclip_torch(rabs_gt, a_min=self.amin, a_max=self.amax, DEBUG=self.DEBUG)  # 1024    
        if self.DEBUG :
            print("DEBUG 9  dataloader :  rabs_gt.shape, rzc_gt.shape ", rabs_gt.shape, rzc_gt.shape)        
           
    
        if self.norm_rabs : 
            rabs_gt    = get_drc_torch(rabs_gt)    
        rabs_gt_out = rabs_gt   
        if self.DEBUG :
            print("DEBUG 10  dataloader :  after drc rabs_gt_out.shape, rabs_gt_out.shape ", rabs_gt.shape, rabs_gt_out.shape)        
                   

        if self.norm_rzc : 
            rzc_gt    = get_drc_torch(rzc_gt)   
        rzc_gt_out = rzc_gt
        
        if self.DEBUG :
            print("DEBUG 11  dataloader :  after drc rzc_gt.shape, rzc_gt_out.shape ", rzc_gt.shape, rzc_gt_out.shape)        
                       
        
        mel_gt_out= torch.squeeze(mel_gt_out, 0)
        if self.DEBUG :
            print("DEBUG 12  dataloader :  after squeeze mel_gt_out ", mel_gt_out.shape)        
                         
        mag_gt_out= torch.squeeze(mag_gt_out, 0)
        if self.DEBUG :
            print("DEBUG 13  dataloader :  after mag_gt_out ",  mag_gt_out.shape)        
                         
        ang_gt= torch.squeeze(ang_gt, 0)        
        if self.DEBUG :
            print("DEBUG 14  dataloader :  after squeeze ang_gt",  ang_gt.shape)        
                         
        mag_lossy_half_out= torch.squeeze(mag_lossy_half_out, 0)
        if self.DEBUG :
            print("DEBUG 15 : dataloader :  after squeeze mag_lossy_half_out",  mag_lossy_half_out.shape)        
                                 
        mag_lossy_full_out= torch.squeeze(mag_lossy_full_out, 0)
        if self.DEBUG :
            print("DEBUG 16 : dataloader :  after drc squeezemag_lossy_full_out ",  mag_lossy_full_out.shape)        
                                 
        rabs_gt_out= torch.squeeze(rabs_gt_out, 0)
        if self.DEBUG :
            print("DEBUG 17  dataloader :  after squeeze rabs_gt_out",  rabs_gt_out.shape)        
                                 
        rsign_gt= torch.squeeze(rsign_gt, 0)
        if self.DEBUG :
            print("DEBUG 18  dataloader :  after squeeze rsign_gt",  rsign_gt.shape)        
                                 
        rzc_gt_out= torch.squeeze(rzc_gt_out, 0)
        if self.DEBUG :
            print("DEBUG 19  dataloader :  after squeeze rzc_gt_out ", rzc_gt_out.shape)        
                                 
        #out = (nframes, nsamples, nsamples_pad, audio.squeeze(1)  , mel_gt_out.squeeze(1), mag_gt_out.squeeze(1), ang_gt.squeeze(1), mag_lossy_half_out.squeeze(1), mag_lossy_full_out.squeeze(1), rabs_gt_out.squeeze(1), rsign_gt.squeeze(1),  rzc_gt_out.squeeze(1))

        out = (nframes, audio.squeeze(1), mel_gt_out.squeeze(1),  mag_lossy_full_out.squeeze(1), rabs_gt_out.squeeze(1), rsign_gt.squeeze(1) )
        if self.DEBUG :
            print("DEBUG 20  dataloader ", len(out) , nframes, audio.shape,  mel_gt_out.shape, mag_lossy_full_out.shape,  rabs_gt_out.shape, rsign_gt.shape  )   
 
        return out

    def __len__(self):
        return len(self.audio_files)       
  
