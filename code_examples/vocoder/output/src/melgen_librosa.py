import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa

from audioutil_librosa import load_wav_librosa, save_wav_librosa, resample_wav_librosa, get_wav_fromfile_librosa
from audioutil_librosa import adjust_wav_librosa, enpad_wav_librosa, depad_wav_librosa, norm_wav_librosa
from audioutil_librosa import get_magphase_librosa, get_magphase_inv_librosa, get_magangle_librosa, get_magangle_inv_librosa
from audioutil_librosa import get_abssign_librosa, get_abssign_inv_librosa, get_zeroclip_librosa,  get_window_librosa
from audioutil_librosa import get_lin2mel_librosa, get_mel2lin_librosa, get_mel_basis_half_librosa, get_mel_basis_full_librosa
from audioutil_librosa import get_drc_librosa, get_drc_inv_librosa
from audioutil_librosa import get_spec_stft_librosa, get_spec_stft_inv_librosa
from audioutil_librosa import get_spec_strft_librosa, get_spec_strft_inv_librosa
from audioutil_librosa import get_spec_stdct_librosa, get_spec_stdct_inv_librosa

from utils import get_dataset_filelists 

from audioutil_torch import get_drc_torch, get_drc_inv_torch
MAX_WAV_VALUE = 32768.0




def get_waveform(pred_input, pair_input,  spec_option=1, 
                    sr=22050,  n_fft=1024,  hop_length=256,  win_type='hann' , a_min = 0.0, a_max=99999,  window_val = None, 
                 norm_mel=True, norm_lossy=True, norm_mag=True, norm_rabs=False, norm_rzc=False,  DEBUG=False):
    
    if spec_option == 1: #1mel_TO_mag  --> wav    (spec_option, mel_gt.shape[1],  wav, mel_gt.T, mag_gt.T )              
        if norm_mag == True : 
            pred_input = get_drc_inv(pred_input)
            
        spec_complex_gt  = get_magangle_inv (pred_input, pair_input, DEBUG=DEBUG).T        
        wav_recon        = get_spec_stft_inv(spec_complex_gt,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 513
        out = wav_recon 
        
    elif spec_option == 2 : #2mag_lossy_TO_mag  (spec_option, mel_gt.shape[1],  wav, mag_lossy_half.T, mag_gt.T )      
        if norm_mag == True : 
            pred_input = get_drc_inv(pred_input)        
        spec_complex_gt = get_magangle_inv (pred_input, pair_input, DEBUG=DEBUG).T        
        wav_recon       = get_spec_stft_inv(spec_complex_gt,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 513
        out = wav_recon      
        
    elif spec_option == 3 :          #3mel_TO_rabs (spec_option, mel_gt.shape[1],  wav, mel_gt.T, rabs_gt.T )
        if norm_rabs == True : 
            pred_input = get_drc_inv(pred_input)        
        spec_real_gt    = get_abssign_inv(pred_input,  pair_input, DEBUG=DEBUG).T         
        wav_recon       = get_spec_strft_inv(spec_real_gt,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 513
        out = wav_recon             

    elif spec_option ==4 : #4mag_lossy_half_TO_rabs     (spec_option, mel_gt.shape[1],  wav, mag_lossy_half.T, rabs_gt.T )        
        if norm_rabs == True : 
            pred_input = get_drc_inv(pred_input)            
        spec_real_gt    = get_abssign_inv(pred_input,  pair_input, DEBUG=DEBUG).T 
        wav_recon       = get_spec_strft_inv(spec_real_gt,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 513
        out = wav_recon                    
        
    elif spec_option == 5 :  #5mag_lossy_full_TO_rabs  (spec_option, rabs_gt.shape[1],  wav,  mag_lossy_full.T, rabs_gt.T )     
        if norm_rabs == True : 
            pred_input = get_drc_inv(pred_input)            
        spec_real_gt    = get_abssign_inv(pred_input,  pair_input, DEBUG=DEBUG).T  # sign        
        wav_recon       = get_spec_strft_inv(spec_real_gt,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 1024
        out = wav_recon          
        
    elif spec_option == 6 : #6rabs_TO_rsign   (spec_option, mel_gt.shape[1],  wav, rabs_gt.T, rsign_gt.T )                  
        if norm_rabs == True : 
            pair_input = get_drc_inv(pair_input)            
        spec_real_gt    =  get_abssign_inv(pred_input,  pair_input, DEBUG=DEBUG).T      
        wav_recon       = get_spec_strft_inv(spec_real_gt,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 1024
        out = wav_recon                                             
        
    elif spec_option == 7:        # 7full   (spec_option, mel_gt.shape[1],  wav, mel_gt.T, mag_gt.T, mag_lossy_half.T, mag_lossy_full.T,  spec_strft_gt.T, rabs_gt.T, rsign_gt.T,  rzc_gt.T)    
        print("select specific spectrogram")
        out = []      
    else :
        print("No implementation for spec_option in gen_spectrogram")
        out=[]

    return out 

    

def get_spectrogram(wav,  spec_option=5, 
                    sr=22050,  n_fft=1024,  hop_length=256,  win_type='hann' , a_min = 0.0, a_max=99999,  
                    n_mels=80, fmin=0, fmax=None, 
                    mel_basis_half=None, mel_basis_full=None, 
                    mel_basis_half_T=None, mel_basis_full_T=None, 
                    window_val = None, 
                    norm_mel=True, norm_lossy=True, norm_mag=True, norm_rabs=False, norm_rzc=False, DEBUG=False):
    import numpy as np
#1mel_TO_mag             (spec_option, mel_gt.shape[1],   mel_gt.T,           mag_gt.T, ang_gt.T )   
#2mag_lossy_TO_mag       (spec_option, mel_gt.shape[1],   mag_lossy_half.T,   mag_gt.T, ang_gt.T )
#3mel_TO_rabs            (spec_option, mel_gt.shape[1],   mel_gt.T,          rabs_gt.T, rsign_gt.T )
#4mag_lossy_half_TO_rabs (spec_option, mel_gt.shape[1],   mag_lossy_half.T,  rabs_gt.T, rsign_gt.T )    
#5mag_lossy_full_TO_rabs (spec_option, rabs_gt.shape[1],  mag_lossy_full.T,  rabs_gt.T, rsign_gt.T ) 
#6rabs_TO_rsign          (spec_option, mel_gt.shape[1],   rabs_gt.T,        rsign_gt.T, rabs_gt.T )   
#7full                   (spec_option, mel_gt.shape[1],   mel_gt.T, mag_gt.T, mag_lossy_half.T, mag_lossy_full.T,  spec_strft_gt.T, rabs_gt.T, rsign_gt.T,  rzc_gt.T)   

    
    if spec_option == 1: #1mel_TO_mag     (spec_option, mel_gt.shape[1],  wav, mel_gt.T, mag_gt.T )   
        spec_stft_gt    = get_spec_stft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 513
        mag_gt, ang_gt  = get_magangle(spec_stft_gt)
        mel_gt          = get_lin2mel(mag_gt, mel_basis = mel_basis_half, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)         
        if norm_mel : 
            mel_gt    = get_drc(mel_gt)       
        mel_gt_out = mel_gt 
        
        if norm_mag : 
            mag_gt    = get_drc(mag_gt)       
        mag_gt_out = mag_gt         
        
        out = (spec_option, mel_gt.shape[1],  mel_gt_out.T, mag_gt_out.T, ang_gt.T ) # spec_option, nframes, input, ref_gt, pair_gt
        
    elif spec_option == 2 : #2mag_lossy_TO_mag  (spec_option, mel_gt.shape[1],  wav, mag_lossy_half.T, mag_gt.T )      
        spec_stft_gt    = get_spec_stft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,  win_type=win_type, center=False, DEBUG=DEBUG) # 513
        mag_gt, ang_gt  = get_magangle(spec_stft_gt)
        mel_gt          = get_lin2mel(mag_gt, mel_basis = mel_basis_half, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)         
        if norm_mel : 
            mel_gt    = get_drc(mel_gt)       
        mel_gt_out = mel_gt    
        if norm_mel : 
            mel_gt   = get_drc_inv(mel_gt) 
        if norm_mag : 
            mag_gt    = get_drc(mag_gt)       
        mag_gt_out = mag_gt              
        mag_lossy_half = get_mel2lin(mel_gt, mel_basis_T = mel_basis_half_T, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)  # 513
        out = (spec_option, mel_gt.shape[1],  mag_lossy_half.T, mag_gt_out.T, ang_gt.T )    # spec_option, nframes, input, ref_gt, pair_gt     
        
    elif spec_option == 3 :          #3mel_TO_rabs (spec_option, mel_gt.shape[1],  wav, mel_gt.T, rabs_gt.T )
        spec_stft_gt    = get_spec_stft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,  win_type=win_type, center=False, DEBUG=DEBUG) # 513
        mag_gt, ang_gt  = get_magangle(spec_stft_gt)
        mel_gt          = get_lin2mel(mag_gt, mel_basis = mel_basis_half, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)         
        if norm_mel : 
            mel_gt    = get_drc(mel_gt)       
        mel_gt_out = mel_gt              
        spec_strft_gt     = get_spec_strft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 1024
        rabs_gt, rsign_gt = get_abssign(spec_strft_gt, DEBUG=DEBUG) # 1024
        
        if norm_rabs : 
            rabs_gt    = get_drc(rabs_gt)       
        rabs_gt_out = rabs_gt         
        

        out = (spec_option, mel_gt.shape[1],  mel_gt_out.T, rabs_gt_out.T, rsign_gt.T ) # spec_option, nframes, input, ref_gt, pair_gt

    elif spec_option ==4 : #4mag_lossy_half_TO_rabs     (spec_option, mel_gt.shape[1],  wav, mag_lossy_half.T, rabs_gt.T )        
        spec_stft_gt    = get_spec_stft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length, win_type=win_type,    center=False, DEBUG=DEBUG) # 513
        mag_gt, ang_gt  = get_magangle(spec_stft_gt)
        mel_gt          = get_lin2mel(mag_gt, mel_basis = mel_basis_half, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)         
        if norm_mel : 
            mel_gt    = get_drc(mel_gt)       
        mel_gt_out = mel_gt    
        if norm_mel : 
            mel_gt   = get_drc_inv(mel_gt)            
        mag_lossy_half = get_mel2lin(mel_gt, mel_basis_T = mel_basis_half_T, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)  # 513          
        if norm_lossy : 
            mag_lossy_half    = get_drc(mag_lossy_half)       
        mag_lossy_half_out = mag_lossy_half  
       
        spec_strft_gt     = get_spec_strft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,    win_type=win_type, center=False, DEBUG=DEBUG) # 1024
        rabs_gt, rsign_gt = get_abssign(spec_strft_gt, DEBUG=DEBUG) # 1024
        if norm_rabs : 
            rabs_gt    = get_drc(rabs_gt)       
        rabs_gt_out = rabs_gt             
        out = (spec_option, mel_gt.shape[1],  mag_lossy_half_out.T, rabs_gt_out.T, rsign_gt.T )     # spec_option, nframes, input, ref_gt, pair_gt            
        
    elif spec_option == 5 :  #5mag_lossy_full_TO_rabs  (spec_option, rabs_gt.shape[1],  wav,  mag_lossy_full.T, rabs_gt.T )     
        spec_stft_gt    = get_spec_stft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,    win_type=win_type, center=False, DEBUG=DEBUG) # 513
        mag_gt, ang_gt  = get_magangle(spec_stft_gt)
        mel_gt          = get_lin2mel(mag_gt, mel_basis = mel_basis_half, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)         
        if norm_mel : 
            mel_gt    = get_drc(mel_gt)       
        mel_gt_out = mel_gt    
        if norm_mel : 
            mel_gt   = get_drc_inv(mel_gt)            
        mag_lossy_full = get_mel2lin(mel_gt, mel_basis_T = mel_basis_full_T, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)  #1024            
        if norm_lossy :     
            mag_lossy_full    = get_drc(mag_lossy_full)            
        mag_lossy_full_out = mag_lossy_full  
        
        spec_strft_gt     = get_spec_strft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 1024
        rabs_gt, rsign_gt = get_abssign(spec_strft_gt, DEBUG=DEBUG) # 1024
        if norm_rabs : 
            rabs_gt    = get_drc(rabs_gt)       
        rabs_gt_out = rabs_gt   
        
        out = (spec_option, mel_gt.shape[1],   mag_lossy_full_out.T, rabs_gt_out.T, rsign_gt.T )     # spec_option, nframes, input, ref_gt, pair_gt    
        
    elif spec_option == 6 : #6rabs_TO_rsign   (spec_option, mel_gt.shape[1],  wav, rabs_gt.T, rsign_gt.T )                  
        spec_strft_gt     = get_spec_strft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,  win_type=win_type, center=False, DEBUG=DEBUG) # 1024
        rabs_gt, rsign_gt = get_abssign(spec_strft_gt, DEBUG=DEBUG) # 1024
        if norm_rabs : 
            rabs_gt    = get_drc(rabs_gt)       
        rabs_gt_out = rabs_gt           
        out = (spec_option, rabs_gt.shape[1],  rabs_gt_out.T, rsign_gt.T , rabs_gt_out.T )     # spec_option, nframes, input, ref_gt, pair_gt                                   
        
    elif spec_option == 7:        # 7full   (spec_option, mel_gt.shape[1],  mel_gt.T, mag_gt.T, mag_lossy_half.T, mag_lossy_full.T,  spec_strft_gt.T, rabs_gt.T, rsign_gt.T,  rzc_gt.T)    
        spec_stft_gt    = get_spec_stft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,   win_type=win_type, center=False, DEBUG=DEBUG) # 513
        mag_gt, ang_gt  = get_magangle(spec_stft_gt)

        mel_gt          = get_lin2mel(mag_gt, mel_basis = mel_basis_half, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)         
        if norm_mel : 
            mel_gt    = get_drc(mel_gt)       
        mel_gt_out = mel_gt    
        if norm_mel : 
            mel_gt   = get_drc_inv(mel_gt)            
        if norm_mag : 
            mag_gt    = get_drc(mag_gt)       
        mag_gt_out = mag_gt  
                    
        mag_lossy_half = get_mel2lin(mel_gt, mel_basis_T = mel_basis_half_T, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)  # 513       
        mag_lossy_full = get_mel2lin(mel_gt, mel_basis_T = mel_basis_full_T, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, DEBUG=DEBUG)  #1024            
        if norm_lossy : 
            mag_lossy_half    = get_drc(mag_lossy_half)       
            mag_lossy_full    = get_drc(mag_lossy_full)             
        mag_lossy_half_out = mag_lossy_half  
        mag_lossy_full_out = mag_lossy_full  
  

        spec_strft_gt     = get_spec_strft(wav,   sr=sr,  n_fft=n_fft,  hop_length=hop_length,  win_type=win_type, center=False, DEBUG=DEBUG) # 1024
        rabs_gt, rsign_gt = get_abssign(spec_strft_gt, DEBUG=DEBUG) # 1024
        rzc_gt            = get_zeroclip(spec_strft_gt, a_min=a_min, a_max=a_max, DEBUG=DEBUG) # 1024
        if norm_rabs : 
            rabs_gt    = get_drc(rabs_gt)       
        rabs_gt_out = rabs_gt   
        
        if norm_rzc : 
            rzc_gt    = get_drc(rzc_gt)       
        rzc_gt_out = rzc_gt
        
        out = (spec_option,  mel_gt.shape[1], mel_gt_out.T, mag_gt_out.T, mag_lossy_half_out.T, mag_lossy_full_out.T,  spec_strft_gt.T, rabs_gt_out.T, rsign_gt.T,  rzc_gt_out.T)
    else :
        print("No implementation for spec_option in gen_spectrogram")
        out=[]

    return out     


class MILKDataset_librosa(torch.utils.data.Dataset):
    def __init__(self, filelists, args, data_config):
            
        
        self.audio_files = filelists        
        random.seed(1234)
        if args.shuffle:
            random.shuffle(self.audio_files)
            
            
        self.device            = args.device   
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
        self.win_length        = data_config.n_fft
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

        self.window_val        = get_window_librosa(win_length=self.n_fft, win_type=self.win_type)
        self.mel_basis_half    = get_mel_basis_half_librosa(      sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin , fmax=self.fmax, DEBUG=self.DEBUG )
        self.mel_basis_half_T  = get_mel_basis_half_librosa(      sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin , fmax=self.fmax, DEBUG=self.DEBUG ).T          
        self.mel_basis_full    = get_mel_basis_full_librosa( sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin , fmax=self.fmax, DEBUG=self.DEBUG )      
        self.mel_basis_full_T  = get_mel_basis_full_librosa( sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin , fmax=self.fmax, DEBUG=self.DEBUG ).T
        if self.DEBUG :
            print("DEBUG : init data loader")
            print("self.window_val       ", self.window_val.shape  )
            print("self.mel_basis_half   ", self.mel_basis_half.shape   )
            print("self.mel_basis_half_T ", self.mel_basis_half_T.shape   )
            print("self.mel_basis_full   ", self.mel_basis_full.shape   )
            print("self.mel_basis_full_T ", self.mel_basis_full_T.shape   )           
            
                 

    def __getitem__(self, index):
        # step1. filename ( get from list) glob
        filepath = self.audio_files[index]
          
        if self.DEBUG :
            print ("DEBUG : idx, file" ,index, filepath )
        # step2. load wav
        audio = get_wav_fromfile(filepath, sr=self.sr, hop_length=self.hop_length,  pad_nums=self.pad_nums, mono=True, dur=15) #normalize, adjust 

        if self.DEBUG :
            print ("DEBUG : len audio ", len(audio) )
            import IPython.display as ipd
            ipd.display(ipd.Audio(audio, rate=self.sr))                                   
            
        #step3. split wav to segment    (numpy)     
        if self.split:
            if self.DEBUG :
                print("DEBUG start split")
                
            if len(audio) >= self.segment_size:
                if self.DEBUG :
                    print ("DEBUG :audio is large")
                    
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
                    print ("DEBUG :audio is small")                
                audio = librosa.util.fix_length(audio, self.segment_size)
                if self.DEBUG :
                    print ("DEBUG : segment  ", len(audio),self.segment_size    )                
                    import IPython.display as ipd
                    ipd.display(ipd.Audio(audio, rate=self.sr))                

        #step4. get mel and other spectrogram
        if self.DEBUG : 
            print ("DEBUG : end of if  " ) 
        out=get_spectrogram(audio,  spec_option=self.spec_option, ### configure it for each cases 
                            sr=self.sr,  n_fft=self.n_fft,  hop_length=self.hop_length, win_type=self.win_type , 
                            a_min =self.amin,  a_max=self.amax, 
                            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
                            norm_mel=self.norm_mel, norm_lossy=self.norm_lossy, norm_mag=self.norm_mag, norm_rabs=self.norm_rabs,  norm_rzc=self.norm_rzc,
                            window_val = self.window_val, 
                            mel_basis_half=self.mel_basis_half, mel_basis_full = self.mel_basis_full, 
                            mel_basis_half_T=self.mel_basis_half_T, mel_basis_full_T = self.mel_basis_full_T, 
                            DEBUG=self.DEBUG  )
        if self.DEBUG :
            print ("DEBUG : get_spectrogram  ", len(out),  )        
        return out

    def __len__(self):
        return len(self.audio_files)
