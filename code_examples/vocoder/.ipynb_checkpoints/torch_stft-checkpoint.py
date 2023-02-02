
import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.util import pad_center, tiny  
from scipy.fftpack import rfft as rfft   
from scipy.fftpack import dct as dct  
from scipy.fftpack import idct as idct 
import time

import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util


def window_sumsquare(window_type, n_frames, hop_length=256, win_length=1024,
                     n_fft=1024, dtype=np.float32, norm=None):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window_type, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x



class STDCT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=None,
                 window_type='hann', pad=True , args=None, DEBUG=False):

        super(STDCT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window_type = window_type
        self.forward_transform = None
        self.pad = pad 
        if self.pad :
            self.pad_amount =  int(self.filter_length / 2)   
        else :
            self.pad_amount = 0
        #
        self.device = torch.device(args.device, args.local_rank)
        self.DEBUG = DEBUG 
        if self.DEBUG:
            print("DEBUG : STDCT init   start " )    
        tic=time.time()

        scale = self.filter_length / self.hop_length
        dct_basis  =  dct(np.eye(self.filter_length), type=2, norm='ortho'  ) # DCT      
        idct_basis = idct(np.eye(self.filter_length), type=2, norm='ortho'  ) # iDCT 
        if self.DEBUG:
            print("DEBUG : STDCT init   dct_basis ", dct_basis.shape)    

        forward_basis = torch.FloatTensor( dct_basis  ) 
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * forward_basis).T  ).to(self.device )         
        #inverse_basis = torch.FloatTensor(idct_basis  ) 
 

        
        
        if self.DEBUG:
            print("DEBUG : STDCT init   forward_basis ", forward_basis.shape)         

        forward_basis= forward_basis.reshape(   forward_basis.size(0), 1, forward_basis.size(-1) ).to(self.device )             
        inverse_basis= inverse_basis.reshape(   inverse_basis.size(0), 1, inverse_basis.size(-1) ).to(self.device )           
        if self.DEBUG:
            print("DEBUG : STDCT init   forward_basis ", forward_basis.shape)           
            
        assert(filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window_type, self.win_length, fftbins=True)
        if self.DEBUG:
            print("DEBUG : STDCT init  fft_window ", fft_window.shape)        
        #fft_window = pad_center(fft_window, filter_length) center=False for chunk stream
        fft_window = torch.from_numpy(fft_window).float().to(self.device ) 
        if self.DEBUG:
            print("DEBUG : STDCT init  fft_window  ", fft_window.shape)        

        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window
        if self.DEBUG:
            print("DEBUG : STDCT init  forward_basis after win mul ", forward_basis.shape)        

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())
        toc = time.time()
        dur = toc-tic
        if self.DEBUG :
            print("DEBUG : STDCT init  {:4.2f} ms ".format(dur*1000) )
            print("DEBUG : STDCT init dct_basis", dct_basis.shape, idct_basis.shape)   
            print("DEBUG : STDCT init forward_basis", forward_basis.shape, inverse_basis.shape)
            print("DEBUG : STDCT init window", fft_window.shape)
            print("DEBUG : STDCT init end")


    def transform(self, input_data):
        input_data = input_data.to(self.device ) 
        if self.DEBUG:
            print("DEBUG : STDCT transform start ")
        tic = time.time()
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        if self.DEBUG:
            print("DEBUG : STDCT transform input_data ", input_data.shape)

        input_data = F.pad( input_data.unsqueeze(1),
                            (self.pad_amount, self.pad_amount , 0, 0), 
                           mode='constant').to(self.device ) 
        if self.DEBUG:
            print("DEBUG : STDCT transform input_data after F.pad", input_data.shape)     
        input_data = input_data.squeeze(1)
        if self.DEBUG:
            print("DEBUG : STDCT transform input_data afater squeeze", input_data.shape)     

        if self.DEBUG:
            print("DEBUG : STDCT transform self.forward_basis", self.forward_basis.shape)   

        if self.DEBUG:
            print("DEBUG : STDCT transform before 1D conv")
        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)
        if self.DEBUG:
            print("DEBUG : after 1D conv")        

        # cutoff = int((self.filter_length / 2) + 1)
        #real_part = forward_transform[:, :cutoff, :]
        #imag_part = forward_transform[:, cutoff:, :]

        if self.DEBUG:
            print("DEBUG : STDCT transform forward_transform", forward_transform.shape)   

        rabs  = torch.abs( forward_transform[:, :None, :]).to(self.device )  
        rsign = torch.sign(forward_transform[:, :None, :]).to(self.device )    

        if self.DEBUG:
            print("DEBUG : STDCT transform rabs", rabs.shape)              
            print("DEBUG : STDCT transform rsign", rsign.shape)   

        toc = time.time()
        dur = toc-tic
        if self.DEBUG:
            print("DEBUG  STDCT transform {:4.2f}ms for {}samples --> {} {} {} frame ".format(dur*1000, input_data.shape, forward_transform.shape, rabs.shape, rsign.shape  )  ) 
        return rabs, rsign

    def inverse(self, rabs, rsign):
        rabs = rabs.to(self.device ) 
        rsign = rsign.to(self.device ) 
        
        if self.DEBUG:
            print("DEBUG : STDCT inverse start ")        
        tic = time.time()
        if self.DEBUG:
            print("DEBUG : STDCT inverse rabs, rsign  ", rabs.shape, rsign.shape )   


        recombine_rabs_rsign =( rabs * torch.sign(rsign) ).to(self.device ) 

        if self.DEBUG:
            print("DEBUG : STDCT inverse recombine_rabs_rsign  ", recombine_rabs_rsign.shape)   

        if self.DEBUG:
            print("DEBUG : STDCT inverse self.inverse_basis  ", self.inverse_basis.shape)   


        if self.DEBUG:
            print("DEBUG : STDCT inverse before conv  ")    

        inverse_transform = F.conv_transpose1d(
            recombine_rabs_rsign,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0).to(self.device ) 
        if self.DEBUG:
            print("DEBUG : STDCT inverse after  conv ")    

        if self.window_type is not None:
            window_sum = window_sumsquare(
                self.window_type, rabs.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32) 
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device).to(self.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices].to(self.device)

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length
        if self.DEBUG:
            print("DEBUG : STDCT inverse after  NOLA ")   

        inverse_transform = inverse_transform[..., self.pad_amount :].to(self.device)
        inverse_transform = inverse_transform[..., : (rabs.size(-1)-1) * self.hop_length].to(self.device)
        inverse_transform = inverse_transform.squeeze(1).to(self.device)
        toc = time.time()
        dur = toc - tic 
        if self.DEBUG:
            print("DEBUG  STDCT inverse {:4.2f}ms for {} {} -->  {} ".format(dur*1000, rabs.shape, rsign.shape, inverse_transform.shape  )  ) 

        return inverse_transform

    def forward(self, input_data):
        input_data=input_data.to(self.device)
        self.rabs, self.rsign = self.transform(input_data).to(self.device)
        reconstruction = self.inverse(self.rabs, self.rsign).to(self.device)
        return reconstruction



class STRFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=None,
                 window_type='hann', pad=True, args=None,  DEBUG=False):

        super(STRFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window_type = window_type
        self.forward_transform = None
        self.pad = pad 
        if self.pad :
            self.pad_amount =   int(self.filter_length / 2)   
        else :
            self.pad_amount = 0
        self.device = torch.device(args.device, args.local_rank)
        self.DEBUG = DEBUG 

        tic=time.time()

        scale = self.filter_length / self.hop_length
        rfft_basis  =  rfft(np.eye(self.filter_length) )  # packed rfft      
        #irfft_basis = irfft(np.eye(self.filter_length)   ) # packed irfft
        if self.DEBUG:
            print("DEBUG : STRFT init rfft_basis ", rfft_basis.shape)    

        forward_basis = torch.FloatTensor( rfft_basis  ).to(self.device)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * rfft_basis).T  ).to(self.device)
        if self.DEBUG:
            print("DEBUG : STRFT init forward_basis ", forward_basis.shape)        

        forward_basis= forward_basis.reshape(   forward_basis.size(0), 1, forward_basis.size(-1) ).to(self.device)
        inverse_basis= inverse_basis.reshape(   inverse_basis.size(0), 1, inverse_basis.size(-1) ).to(self.device)       
        if self.DEBUG:
            print("DEBUG : STRFT init  forward_basis ", forward_basis.shape)           
            
        assert(filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window_type, self.win_length, fftbins=True)
        if self.DEBUG:
            print("DEBUG : STRFT init fft_window ", fft_window.shape)        
        #fft_window = pad_center(fft_window, filter_length) center=False for chunk stream
        fft_window = torch.from_numpy(fft_window).float().to(self.device)
        if self.DEBUG:
            print("DEBUG : STRFT init fft_window  ", fft_window.shape)        

        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window
        if self.DEBUG:
            print("DEBUG : STRFT forward_basis after win mul ", forward_basis.shape)        

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())
        toc = time.time()
        dur = toc-tic
        if self.DEBUG :
            print("DEBUG : STRFT init {:4.2f} ms ".format(dur*1000) )
            print("DEBUG : STRFT init rfft_basis", rfft_basis.shape )   
            print("DEBUG : STRFT init forward_basis", forward_basis.shape, inverse_basis.shape)
            print("DEBUG : STRFT init window", fft_window.shape)
            print("DEBUG : STRFT init end")


    def transform(self, input_data):
        input_data = input_data.to(self.device)
        if self.DEBUG:
            print("DEBUG : stdct transform start ")
        tic = time.time()
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples).to(device)
        if self.DEBUG:
            print("DEBUG : STRFT transform input_data ", input_data.shape)

        input_data = F.pad( input_data.unsqueeze(1),
                           (self.pad_amount, self.pad_amount, 0, 0),
                           mode='constant').to(self.device)
        if self.DEBUG:
            print("DEBUG : STRFT transform input_data after F.pad", input_data.shape)     
        input_data = input_data.squeeze(1).to(self.device)
        if self.DEBUG:
            print("DEBUG : STRFT transform input_data afater squeeze", input_data.shape)     

        if self.DEBUG:
            print("DEBUG : STRFT transform self.forward_basis", self.forward_basis.shape)   

        if self.DEBUG:
            print("DEBUG : STRFT transform before 1D conv")
        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0).to(self.device)
        if self.DEBUG:
            print("DEBUG : after 1D conv")        

        # cutoff = int((self.filter_length / 2) + 1)
        #real_part = forward_transform[:, :cutoff, :]
        #imag_part = forward_transform[:, cutoff:, :]

        if self.DEBUG:
            print("DEBUG : STRFT transform forward_transform", forward_transform.shape)   

        rabs  = torch.abs( forward_transform[:, :None, :]).to(self.device)
        rsign = torch.sign(forward_transform[:, :None, :]).to(self.device)

        if self.DEBUG:
            print("DEBUG : STRFT transform rabs", rabs.shape)              
            print("DEBUG : STRFT transform rsign", rsign.shape)   

        toc = time.time()
        dur = toc-tic
        if self.DEBUG:
            print("DEBUG  STRFT transform {:4.2f}ms for {}samples --> {} {} {} frame ".format(dur*1000, input_data.shape, forward_transform.shape, rabs.shape, rsign.shape  )  ) 
        return rabs, rsign

    def inverse(self, rabs, rsign):
        rabs=rabs.to(self.device)
        rsign=rsign.to(self.device)
        if self.DEBUG:
            print("DEBUG : STRFT inverse start ")        
        tic = time.time()
        if self.DEBUG:
            print("DEBUG : STRFT inverse rabs, rsign  ", rabs.shape, rsign.shape )   


        recombine_rabs_rsign = (rabs * torch.sign(rsign)  ).to(self.device)

        if self.DEBUG:
            print("DEBUG : STRFT inverse recombine_rabs_rsign  ", recombine_rabs_rsign.shape)   

        if self.DEBUG:
            print("DEBUG : STRFT inverse inverse self.inverse_basis  ", self.inverse_basis.shape)   


        if self.DEBUG:
            print("DEBUG : STRFT inverse before conv  ")    

        inverse_transform = F.conv_transpose1d(
            recombine_rabs_rsign,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0).to(self.device)
        if self.DEBUG:
            print("DEBUG : STRFT inverse after  conv ")    

        if self.window_type is not None:
            window_sum = window_sumsquare(
                self.window_type, rabs.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32) 
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices].to(self.device)

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length
        if self.DEBUG:
            print("DEBUG : STRFT inverse after  NOLA ")   

        inverse_transform = inverse_transform[..., self.pad_amount :].to(self.device)
        inverse_transform = inverse_transform[..., : (rabs.size(-1) -1)* self.hop_length ].to(self.device)
        inverse_transform = inverse_transform.squeeze(1).to(self.device)
        toc = time.time()
        dur = toc - tic 
        if self.DEBUG:
            print("DEBUG  STRFT inverse {:4.2f}ms for {} {} -->  {} ".format(dur*1000, rabs.shape, rsign.shape, inverse_transform.shape  )  ) 

        return inverse_transform

    def forward(self, input_data):
        input_data = input_data.to(device)
        self.rabs, self.rsign = self.transform(input_data).to(self.device)
        reconstruction = self.inverse(self.rabs, self.rsign).to(self.device)
        return reconstruction

 
class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window_type='hann', pad=True, args=None, DEBUG=False):

        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window_type = window_type
        self.forward_transform = None
        self.pad = pad 
        if self.pad :
            self.pad_amount =  int(self.filter_length / 2)   
        else : 
            self.pad_amount = 0
        self.device=torch.device(args.device, args.local_rank)
        self.DEBUG = DEBUG

        if self.DEBUG:
            print("DEBUG : init start")
        tic = time.time()
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length)) 
        if self.DEBUG:
            print("DEBUG fourier_basis ", fourier_basis.shape)

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        if self.DEBUG:
            print("DEBUG fourier_basis after cutoff ", fourier_basis.shape)

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :]).to(self.device)
        if self.DEBUG:
            print("DEBUG forward_basis ", forward_basis.shape)
                
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]).to(self.device)
        if self.DEBUG:
            print("DEBUG inverse_basis ", inverse_basis.shape)    

        assert(filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window_type, self.win_length, fftbins=True)
        if self.DEBUG:
            print("DEBUG fft_window ", fft_window.shape)            
        fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float().to(self.device)
        if self.DEBUG:
            print("DEBUG fft_window after center ", fft_window.shape)    
        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window

        if self.DEBUG:
            print("DEBUG forward_basis after win product ", forward_basis.shape)   
            print("DEBUG inverse_basis after win product ", inverse_basis.shape)               

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())
        toc = time.time()
        dur = toc-tic
        if self.DEBUG:
            print("DEBUG : STFT init {:4.2f} ms".format(dur*1000) )        


    def transform(self, input_data):
        input_data = input_data.to(self.device)
        if self.DEBUG:
            print("DEBUG : transform start")
        tic = time.time()        
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples).to(self.device)

        if self.DEBUG:
            print("DEBUG : input_data", input_data.shape)

        input_data = F.pad( input_data.unsqueeze(1),
                           ( self.pad_amount, self.pad_amount, 0, 0),
                           mode='constant').to(self.device)
        
        if self.DEBUG:
            print("DEBUG : input_data after F.pad", input_data.shape)

        input_data = input_data.squeeze(1).to(self.device)
        if self.DEBUG:
            print("DEBUG : input_data after squeeze", input_data.shape)
        if self.DEBUG:
            print("DEBUG : self.forward_basis", self.forward_basis.shape)
        if self.DEBUG:
            print("DEBUG : conv before")

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0).to(self.device)
        if self.DEBUG:
            print("DEBUG : conv after")

        if self.DEBUG:
            print("DEBUG : forward_transform",forward_transform.shape)                    

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :].to(self.device)
        if self.DEBUG:
            print("DEBUG : real_part after cutoff",real_part.shape)                    

        imag_part = forward_transform[:, cutoff:, :].to(self.device)
        if self.DEBUG:
            print("DEBUG : imag_part ",imag_part.shape)                    

        magnitude = torch.sqrt(real_part**2 + imag_part**2).to(self.device)
        if self.DEBUG:
            print("DEBUG : magnitude  ",magnitude.shape)                    

        phase = torch.atan2(imag_part.data, real_part.data).to(self.device)
        if self.DEBUG:
            print("DEBUG : phase   ",phase.shape)                    

        toc = time.time()
        dur =  toc-tic
        if self.DEBUG:
            print("DEBUG : STFT tranform {}ms".format(dur*1000) )        


        return magnitude, phase

    def inverse(self, magnitude, phase):
        magnitude=magnitude.to(self.device)
        phase = phase.to(self.device)
        if self.DEBUG:
            print("DEBUG : inverse start")
        tic = time.time()          
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1).to(self.device)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0).to(self.device)

        if self.window_type is not None:
            window_sum = window_sumsquare(
                self.window_type, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32) 
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]).to(self.device)
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device).to(self.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices].to(self.device)

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount :].to(self.device)
        inverse_transform = inverse_transform[..., :  (magnitude.size(-1) -1) * self.hop_length ].to(self.device)
        inverse_transform = inverse_transform.squeeze(1).to(self.device)

        toc = time.time()
        dur = toc - tic
        if self.DEBUG:
            print("DEBUG : STFT inverse {}ms".format(dur*1000) )        

        return inverse_transform

    def forward(self, input_data):
        input_data = input_data
        self.magnitude, self.phase = self.transform(input_data).to(self.device)
        reconstruction = self.inverse(self.magnitude, self.phase).to(self.device)          
        return reconstruction
