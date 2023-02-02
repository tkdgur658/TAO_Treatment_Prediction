
import torch

import torch.nn as nn
import torch.functional as F
from random import random
from torch.autograd import Variable
from functools import partial

from ffmixer import FFMixer 

class MILK_RNN_1stage(nn.Module):
    def __init__(self, 
                 mel_dim=80, 
                 K = 16, 
                 linear_dim=1024):
        super(MILK_RNN_1stage, self).__init__()
        self.mel_dim = mel_dim #80
        self.linear_dim = linear_dim # 1024

        self.rnnstage = CBHG(mel_dim, K=K, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

    def forward(self, inputs, DEBUG=False):
        # Post net processing below
        nframes,  out  = inputs  # (B, T, nbin)
        out = self.rnnstage(out)
        out = self.last_linear(out)
        return out 
    def infer(self, inputs, DEBUG=False):    
        return self.forward(inputs, DEBUG=DEBUG)          
    
    
class MILK_FFTr_NSE(nn.Module):
    def __init__(self, args, args_data, args_model):
        super(MILK_FFTr_NSE, self).__init__()
        self.norm_mel   = args_data.norm_mel
        self.norm_mag   = args_data.norm_mag
        self.norm_lossy = args_data.norm_lossy
        self.norm_rabs  = args_data.norm_rabs
        self.norm_rzc   = args_data.norm_rzc
        self.DEBUG      = args.DEBUG 
        self.nframes    = int(args_data.max_nframes + 1 ) 
        
        self.fftstage = FFTransformer(
            n_layer     = args_model.nse_nlayers,   
            n_head      = 4,     
            d_model     = args_model.nse_dmodel,     
            d_head      = int(args_model.nse_dmodel/4),     
            d_inner     = int(args_model.nse_dmodel*args_model.nse_expansion) ,  
            kernel_size = 3,  
            dropout     = args_model.nse_dropout,   
            dropatt     = args_model.nse_dropout,    
            dropemb     = args_model.nse_dropout,   
            embed_input = False,
            d_embed     = args_model.nse_input_channels    
        )        
        self.proj_pre   = nn.Linear( args_model.nse_input_channels,  args_model.nse_dmodel, bias=False) 
        self.proj_post  = nn.Linear( args_model.nse_dmodel,  args_model.nse_output_channels, bias=False)                    

    def forward(self, inputs, DEBUG=False):
        # inputs    : nframes, input_data(batch, nframes, nbins)
        # outputs   : nframes, output_data(batch, nframes, nbins)
        nframes, out = inputs            
        out = self.proj_pre(out)
        out, lin_mask = self.fftstage(out, nframes)        
        out = self.proj_post(out)                                     
        return out 
    
    def infer(self, inputs, , DEBUG=False):    
        return self.forward(inputs, DEBUG=DEBUG)      
    

    
class MILK_FFTr_NSR(nn.Module):
    def __init__(self, args, args_data, args_model):
        super(MILK_FFTr_NSE, self).__init__()
        self.norm_mel   = args_data.norm_mel
        self.norm_mag   = args_data.norm_mag
        self.norm_lossy = args_data.norm_lossy
        self.norm_rabs  = args_data.norm_rabs
        self.norm_rzc   = args_data.norm_rzc
        self.DEBUG      = args.DEBUG 
        self.nframes    = int(args_data.max_nframes + 1 ) 
        
        self.fftstage = FFTransformer(
            n_layer     = args_model.nsr_nlayers,   
            n_head      = 4,     
            d_model     = args_model.nsr_dmodel,     
            d_head      = int(args_model.nsr_dmodel/4),     
            d_inner     = int(args_model.nsr_dmodel*args_model.nsr_expansion) ,  
            kernel_size = 3,  
            dropout     = args_model.nsr_dropout,   
            dropatt     = args_model.nsr_dropout,    
            dropemb     = 0.0,   
            embed_input = False,
            d_embed     = args_model.nsr_input_channels    
        )        
        self.proj_pre   = nn.Linear( args_model.nsr_input_channels,  args_model.nsr_dmodel, bias=False) 
        self.proj_post  = nn.Linear( args_model.nsr_dmodel,  args_model.nsr_output_channels, bias=False)                    

    def forward(self, inputs, DEBUG=False):
        # inputs    : nframes, input_data(batch, nframes, nbins)
        # outputs   : nframes, output_data(batch, nframes, nbins)
        nframes, out = inputs            
        out = self.proj_pre(out)
        out, lin_mask = self.fftstage(out, nframes)        
        out = self.proj_post(out)                                     
        return out 
    
    def infer(self, inputs, , DEBUG=False):    
        return self.forward(inputs, DEBUG=DEBUG)      
    

class MILK_FFMixer_NSR(nn.Module):
    def __init__(self, args, args_data, args_model ):
        super(MILK_FFMixer_NSR, self).__init__()
        self.norm_mel   = args_data.norm_mel
        self.norm_mag   = args_data.norm_mag
        self.norm_lossy = args_data.norm_lossy
        self.norm_rabs  = args_data.norm_rabs
        self.norm_rzc   = args_data.norm_rzc
        self.DEBUG      = args.DEBUG 
        
        self.nsr = FFMixer(input_dim  = args_model.nsr_input_channels, 
                           output_dim = args_model.nsr_output_channels, 
                           n_layers   = args_model.nsr_nlayers , 
                           d_model    = args_model.nsr_dmodel, 
                           n_tokens   = int(args_data.max_nframes + 1 ) , 
                           expansion  = args_model.nsr_expansion, 
                           drop       = args_model.nsr_dropout , 
                           fn1        = args_model.nsr_fn1 , 
                           fn2        = args_model.nsr_fn2 , 
                           act1       = args_model.nsr_actfn1, 
                           act2       = args_model.nsr_actfn2,
                           DEBUG = args.DEBUG )                        

    def forward(self, inputs, DEBUG=False):
        # inputs    : (batch, nframes, nbins)  
        # outputs   : (batch, nframes, nbins)
        nframes, out = inputs
        out = self.nsr(out ) 
        return   out
    
    def infer(self, inputs, DEBUG=False):   
        return  self.forward(inputs, DEBUG=DEBUG )  

class MILK_FFMixer_NSE(nn.Module):
    def __init__(self, args, args_data, args_model ):
        super(MILK_FFMixer_NSE, self).__init__()
        self.norm_mel   = args_data.norm_mel
        self.norm_mag   = args_data.norm_mag
        self.norm_lossy = args_data.norm_lossy
        self.norm_rabs  = args_data.norm_rabs
        self.norm_rzc   = args_data.norm_rzc
        self.DEBUG      = args.DEBUG 
        
        self.nse = FFMixer(input_dim  = args_model.nse_input_channels, 
                           output_dim = args_model.nse_output_channels, 
                           n_layers   = args_model.nse_nlayers , 
                           d_model    = args_model.nse_dmodel, 
                           n_tokens   = int(args_data.max_nframes + 1 ) , 
                           expansion  = args_model.nse_expansion, 
                           drop       = args_model.nse_dropout , 
                           fn1        = args_model.nse_fn1 , 
                           fn2        = args_model.nse_fn2 , 
                           act1       = args_model.nse_actfn1, 
                           act2       = args_model.nse_actfn2,
                           DEBUG = args.DEBUG )                     

    def forward(self, inputs, DEBUG=False):
        # inputs    : (batch, nframes, nbins)  
        # outputs   : (batch, nframes, nbins)
        nframes, out = inputs
        out = self.nse(out ) 
        return out
    
    
    def infer(self, inputs, DEBUG=False):          
        return  self.forward(inputs, DEBUG=DEBUG )  

class MILK_FFMixer_shared(nn.Module):
    def __init__(self, args, args_data, args_model ):
        super(MILK_FFMixer_shared, self).__init__()
        self.norm_mel   = args_data.norm_mel
        self.norm_mag   = args_data.norm_mag
        self.norm_lossy = args_data.norm_lossy
        self.norm_rabs  = args_data.norm_rabs
        self.norm_rzc   = args_data.norm_rzc
        self.DEBUG      = args.DEBUG

        self.enc = FFMixer(input_dim  = args_model.enc_input_channels, 
                           output_dim = args_model.enc_output_channels, 
                           n_layers   = args_model.enc_nlayers , 
                           d_model    = args_model.enc_dmodel, 
                           n_tokens   = int(args_data.max_nframes + 1 ) , 
                           expansion  = args_model.enc_expansion, 
                           drop       = args_model.enc_dropout , 
                           fn1        = args_model.enc_fn1 , 
                           fn2        = args_model.enc_fn2 , 
                           act1       = args_model.enc_actfn1, 
                           act2       = args_model.enc_actfn2,
                           DEBUG = args.DEBUG )    
        
        
        self.nsr = FFMixer(input_dim  = args_model.nsr_input_channels, 
                           output_dim = args_model.nsr_output_channels, 
                           n_layers   = args_model.nsr_nlayers , 
                           d_model    = args_model.nsr_dmodel, 
                           n_tokens   = int(args_data.max_nframes + 1 ) , 
                           expansion  = args_model.nsr_expansion, 
                           drop       = args_model.nsr_dropout , 
                           fn1        = args_model.nsr_fn1 , 
                           fn2        = args_model.nsr_fn2 , 
                           act1       = args_model.nsr_actfn1, 
                           act2       = args_model.nsr_actfn2,
                           DEBUG = args.DEBUG )        

        self.nse = FFMixer(input_dim  = args_model.nse_input_channels, 
                           output_dim = args_model.nse_output_channels, 
                           n_layers   = args_model.nse_nlayers , 
                           d_model    = args_model.nse_dmodel, 
                           n_tokens   = int(args_data.max_nframes + 1 ) , 
                           expansion  = args_model.nse_expansion, 
                           drop       = args_model.nse_dropout , 
                           fn1        = args_model.nse_fn1 , 
                           fn2        = args_model.nse_fn2 , 
                           act1       = args_model.nse_actfn1, 
                           act2       = args_model.nse_actfn2,
                           DEBUG = args.DEBUG )                              

    def forward(self, inputs, DEBUG=False):
        # inputs    : (batch, nframes, nbins)  
        # outputs   : (batch, nframes, nbins)  
        nframes, out = inputs
        out = self.enc(out )        
        out1 = self.nsr(out)
        out2 = self.nse(out)
        return  out1, out2
    
    def infer(self, inputs, DEBUG=False):           
        return  self.forward(self, inputs, DEBUG=DEBUG)
