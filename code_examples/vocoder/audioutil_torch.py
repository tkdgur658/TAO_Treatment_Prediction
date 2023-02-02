
import os
import sys
import time 
import librosa
import numpy as np
import scipy
import wave
import IPython.display as ipd 
import matplotlib.pyplot as plt
import soundfile as sf 
import torch 

### all return to torch 
 
#pass 
def load_wav_torch(filepath, sr=22050, mono=True, dur=None, DEBUG=False):
    import librosa
    import numpy 
    import time 
    tic = time.time()
    wav, sr = librosa.core.load(filepath,  sr=sr, mono=mono, res_type='polyphase')
    if (dur == None):
        wav_final   = wav
    else:
        wav_final   = wav[0: int(dur*sr) ]
    n_samples = len(wav_final)
    wav_final = wav_final.astype("float32")
    toc = time.time()
    dur = toc-tic
    if DEBUG :
        print( "DEBUG : {:>6.3f} ms load_wav               : sr{} {:>6.2f} sec audio with {} samples  "
        .format(dur*1000, sr,  len(wav)/sr, n_samples  )  )    
    return torch.FloatTensor(wav_final), sr, n_samples, len(wav)/sr    

def adjust_wav_torch(wav,  sr=22050, hop_length = 256, DEBUG=False ):
    import time
    tic = time.time()    
    #stft/istft truncate  n * hop + remains --> h* hop 
    frames_org = len(wav)/hop_length
    n_frames = int(len(wav) / hop_length  ) + 1 
    samples = int( len(wav)/hop_length)*hop_length
    wav_adj = wav[:samples] 
    frames_adj = len(wav_adj)/hop_length
    toc = time.time()
    dur = toc - tic
    if DEBUG :
        print( "DEBUG : {:>6.3f} ms adjust_wav             : sr{} {:>6.2f} sec audio with {} samples {:>8.1f}frames -->  {} samples {:>8.1f} frames ".
              format(dur*1000, sr,  len(wav)/sr, len(wav), frames_org, len(wav_adj), frames_adj  )  )
    return torch.FloatTensor(wav_adj )

def enpad_wav_torch(wav, hop_length = 256, pad_nums=0 , DEBUG=False ):
    import librosa 
    import time
    tic = time.time()    
    wav_expand = librosa.util.pad_center(wav,   len(wav) + int(2*pad_nums * hop_length), mode='constant')
    toc = time.time()
    dur = toc-tic     
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms enpad_wav              : {} -- {} +{}--> {}  {} "
        .format( dur, len(wav), hop_length, pad_nums, len(wav_expand), len(wav_expand)- len(wav)) )    
    return torch.FloatTensor(wav_expand)

def depad_wav_torch(wav, sr=22050, hop_length = 256, pad_nums=0, DEBUG=False ):
    import time
    tic = time.time()        
    front_pad =     pad_nums * hop_length 
    back_pad  =     pad_nums * hop_length 
    all_pad   = 2 * pad_nums * hop_length 
    wav_trim  = wav[front_pad: len(wav)- back_pad ] 
    toc = time.time()
    dur = toc-tic        
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms depad_wav              : {} -- {} -{}--> {}  {} "
        .format( dur, len(wav), hop_size, pad_nums, len(wav_trim), len(wav_trim)- len(wav)) )    
    return torch.FloatTensor(wav_trim  )

def norm_wav_torch(wav, sr=22050, ratio=0.94 , DEBUG=False):
    import librosa
    import time
    tic = time.time()
    wav_norm = librosa.util.normalize(wav) * ratio   
    toc = time.time()
    dur = toc-tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms norm_wav               : sr{} {:>6.2f} sec audio with {} samples with {}"
        .format(dur * 1000 , sr,  len(wav)/sr, len(wav_norm), ratio )  )   
    return torch.FloatTensor(wav_norm)

def save_wav_torch(filepath, wav, sr=22050, subtype='PCM_16', DEBUG=False):
    import soundfile as sf
    import time
    tic = time.time()
    wav = output.cpu().data.numpy()[..., :]
    sf.write(filepath, wav, sr, subtype=subtype)
    toc = time.time()
    dur = toc - tic
    if DEBUG :
        print( "DEBUG : {:>6.3f} ms save_wav               : sr{} {:>6.2f} sec audio with {} samples format {} "
        .format(dur*1000, sr,  len(wav)/sr, len(wav) , subtype )  )

def resample_wav_torch(wav, org_sr, tgt_sr, res_type = 'polyphase' , DEBUG=False ): #########TODEO
    import librosa 
    import time
    tic = time.time()
    wav_resample = librosa.resample(wav, orig_sr=org_sr, target_sr = tgt_sr, res_type=res_type) 
    toc = time.time()
    dur = toc - tic
    if DEBUG :
        print( "DEBUG : warning. it use librosa {:>6.3f} ms resample_wav           : sr{} {:>6.2f} sec audio with {} samples --{}--> sr{} {:>6.2f} sec audio with {} samples".
              format(dur*1000, org_sr,  len(wav)/org_sr , len(wav),   res_type, tgt_sr, len(wav_resample)/tgt_sr ,len(wav_resample)) )
    return wav_resample        

def get_magangle_torch(spectrogram, DEBUG=False ):
    import time 
    import torch    
    tic = time.time()    
    magnitude   = torch.abs(spectrogram)
    angle       = torch.angle(spectrogram)
    toc = time.time()
    dur = toc - tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_magangle           : {} bin x {} frames   "
        .format( dur*1000,  spectrogram.shape[0], spectrogram.shape[1]  ) )             
    return magnitude, angle

def get_magangle_inv_torch(magnitude, angle,  DEBUG=False ):
    import time   
    tic = time.time()    
    spectrogram = magnitude   * torch.exp( 1j* angle) 
    toc = time.time()
    dur = toc - tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_magangle_inv       : {} bin x {} frames   "
        .format( dur*1000,  spectrogram.shape[0], spectrogram.shape[1]  ) )             
    return spectrogram

def get_magphase_torch(spectrogram, DEBUG=False ):
    import time 
    import torch    
    tic = time.time()    
    magnitude   = torch.abs(spectrogram)
    phase       = torch.exp(1.j* torch.angle(spectrogram) )
    toc = time.time()
    dur = toc - tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_magphase           : {} bin x {} frames   "
        .format( dur*1000,  spectrogram.shape[0], spectrogram.shape[1]  ) )                   
    return magnitude, phase

def get_magphase_inv_torch(magnitude, phase, DEBUG=False ):
    import time 
    tic = time.time()    
    spectrogram       = magnitude * phase
    toc = time.time()
    dur = toc - tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_magphase_inv       : {} bin x {} frames   "
        .format( dur*1000,  spectrogram.shape[0], spectrogram.shape[1]  ) )                   
    return spectrogram

def get_abssign_torch(spectrogram, DEBUG=False ):
    import time 
    import torch 
    tic = time.time()    
    magnitude   = torch.abs(spectrogram)
    signvalue   = torch.sign(spectrogram)
    toc = time.time()
    dur = toc - tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_magsign            : {} bin x {} frames   "
        .format( dur*1000,  spectrogram.shape[0], spectrogram.shape[1]  ) )               
    return magnitude, signvalue

def get_abssign_inv_torch(magnitude, signvalue, DEBUG=False ):   
    import time     
    tic = time.time()    
    spectrogram   = magnitude * signvalue
    toc = time.time()
    dur = toc - tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_abssign_inv        : {} bin x {} frames   "
        .format( dur*1000,  spectrogram.shape[0], spectrogram.shape[1]  ) )               
    return spectrogram

def get_window_torch(win_length=512, win_type = 'hanning'  ):
    from scipy import signal 
    import numpy as np     
 
    if win_type=='hanning' :
        window = np.hanning
        win = window(win_length)
    elif win_type=='boxcar':
        window = signal.boxcar        
        win = window(win_length)
    else :
        window = np.hanning
        win = window(win_length)       
    return torch.FloatTensor(win) 

def get_lin2mel_full_torch(mag , mel_basis, sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None, DEBUG=False ):
    import time
 
    tic = time.time()
    import torch
    mel = torch.matmul(mel_basis, mag)
    toc = time.time()
    dur = toc-tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_lin2mel          : {}  to {} ".format(dur * 1000, mag.shape , mel.shape  )          )   
    return mel

def get_lin2mel_half_torch(mag , mel_basis, sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None, DEBUG=False ):
    import time
   
    tic = time.time()
    import torch
    mel = torch.matmul(mel_basis, mag)
    toc = time.time()
    dur = toc-tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_lin2mel          : {}  to {} ".format(dur * 1000, mag.shape , mel.shape  )          )   
    return mel

def get_mel2lin_half_torch(mel , mel_basis_T, sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None, DEBUG=False):
    import time
    import torch
   
    tic = time.time() 
    mag = torch.matmul(mel_basis_T, mel)
    toc = time.time()
    dur = toc-tic
    if DEBUG :      
        print("DEBUG : {:>6.3f} ms get_mel2lin          : {}  to {} ".format(dur * 1000, mel.shape , mag.shape  )          )   
    return mag

def get_mel2lin_full_torch(mel , mel_basis_T, sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None, DEBUG=False):
    import time
    import torch

    #if DEBUG :
    #    print("DEBUG : mel_basis{} mag{}".format(mel_basis_T.shape, mel.shape) )    
    tic = time.time() 
    mag = torch.matmul(mel_basis_T, mel)
    toc = time.time()
    dur = toc-tic
    if DEBUG :      
        print("DEBUG : {:>6.3f} ms get_mel2lin          : {}  to {} ".format(dur * 1000, mel.shape , mag.shape  )          )   
    return mag


def get_mel_basis_half_torch(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None, DEBUG=False ):
    import librosa
    import time
    tic = time.time()
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    toc = time.time()
    dur = toc-tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_mel_basis        : {}   ".format(dur * 1000, mel_basis.shape )          )   
    return torch.FloatTensor(mel_basis)

def get_mel_basis_full_torch(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None, norm=1, DEBUG=False):
    import time 
    import librosa
    import numpy as np 
    tic = time.time()
    #two_side float
    if fmax is None:
        fmax = int(float(sr) /2)   #  
    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))              
    n_mels   = int(n_mels)
    weights  = np.zeros( (n_mels, int(n_fft+1)) , dtype=float)  
    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft*2)  # bug fixed 2021. 11.1   librosa.fft_frequencies(sr=sr, n_fft=n_fft*2)*2 was bug
    mel_f    = librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)
    fdiff = np.diff(mel_f) 
    ramps = np.subtract.outer(mel_f, fftfreqs)   
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]        
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    weights = np.delete(weights,int(n_fft),1) 
    toc = time.time()
    dur = toc-tic
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_mel_basis        : {}   "
        .format(dur * 1000, weights.shape )          )   

    return torch.FloatTensor(weights)

def get_drc_torch(x, C=1, clip_val=1e-5 ,DEBUG=False):
    dcr = torch.log(torch.clamp(x, min=clip_val) * C)
    return dcr

def get_drc_inv_torch(x, C=1 ,DEBUG=False):
    dcd = torch.exp(x) / C
    return dcd


def get_zeroclip_torch(input, a_min=0, a_max=99999, DEBUG=False):
    import numpy as np
    import time
    tic = time.time()    
    clip_val = torch.clip(input, min=a_min, max=a_max)
    toc = time.time()
    dur = toc - tic 
    if DEBUG : 
        print("DEBUG : {:>6.3f} ms get_zeroclip               : {} "
        .format( dur*1000,  clip_val.shape  ) )        
    return clip_val
