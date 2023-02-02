
import torch

import torch.nn as nn
import torch.functional as F
from random import random
from torch.autograd import Variable
from functools import partial

class FFMixer(nn.Module):
    def __init__(self, 
                 input_dim = 7, 
                 output_dim = 11,
                 n_layers=12, 
                 d_model=8, 
                 n_tokens=29, 
                 expansion=2, 
                 drop=0.,
                 fn1='dense', 
                 fn2='dense', 
                 kernel_size=1, 
                 act1='gelu', 
                 act2='gelu',
                DEBUG=False):  
        super(FFMixer, self).__init__()        
        self.projection_pre =   nn.Linear(input_dim, d_model)
        self.projection_post =  nn.Linear(d_model, output_dim)
        self.n_layers = n_layers

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_tokens= n_tokens
        self.expansion = expansion
        self.fn1=fn1
        self.nf2=fn2
        self.drop=drop
        self.DEBUG = DEBUG        

        self.layer = CoreBlock(dim=d_model, n_tokens=n_tokens, expansion=expansion, drop=drop, fn1=fn1, fn2=fn2, kernel_size=kernel_size, act1=act1, act2=act2, DEBUG=self.DEBUG)
        self.layers = nn.ModuleList([ CoreBlock(dim=d_model, n_tokens=n_tokens, expansion=expansion, drop=drop, fn1=fn1, fn2=fn2, kernel_size=kernel_size, act1=act1, act2=act2, DEBUG=self.DEBUG) for _ in range(self.n_layers)])
                
    def forward(self, x, DEBUG=False):
        if DEBUG: 
            print("DEBUG Mixer: init x ", x.shape)
        out = self.projection_pre(x)
        if DEBUG: 
            print("DEBUG Mixer: after pre ", out.shape)
        for i, layer in enumerate(self.layers):
            out = layer(out, DEBUG=self.DEBUG)
            if DEBUG: 
                print("DEBUG Mixer: loop ",i,  out.shape)            
        if DEBUG: 
            print("DEBUG Mixer: after layers ", out.shape)
        out = self.projection_post(out)
        if DEBUG: 
            print("DEBUG Mixer: after post ", out.shape)
        return out            

class CoreBlock(nn.Module):
    def __init__(self, 
                 dim=8, 
                 n_tokens=21, 
                 expansion=2, 
                 drop=0., 
                 fn1='dense', 
                 fn2='dense', 
                 kernel_size=1, 
                 act1='gelu', 
                 act2='gelu',
                 DEBUG=False): # 197 = 16**2 + 1
        super(CoreBlock, self).__init__()
        self.DEBUG = DEBUG
        # FF over channel(features)
        self.corenet1 = CoreNet(in_features=dim, 
                                expansion=expansion, 
                                fn=fn1,
                                kernel_size=kernel_size, 
                                act_layer=act1, 
                                drop=drop,
                                DEBUG=self.DEBUG)            
        self.norm1 = nn.LayerNorm(dim)
        # FF over position(frame)
        self.corenet2 = CoreNet(in_features=n_tokens, 
                                expansion=expansion, 
                                fn=fn2, 
                                kernel_size=kernel_size,
                                act_layer=act2, 
                                drop=drop,
                                DEBUG=self.DEBUG)
        self.norm2 = nn.LayerNorm(n_tokens)

    def forward(self, x, DEBUG=False):
        if DEBUG:
            print("  DEBUG CoreBlock:  init x          ", x.shape)
        x = x + self.corenet1(self.norm1(x), DEBUG=self.DEBUG)
        if DEBUG:
            print("  DEBUG CoreBlock:  after run       ", x.shape)        
        x = x.transpose(-2, -1)
        if DEBUG:
            print("  DEBUG CoreBlock:  after transpose ", x.shape)   
        x = x + self.corenet2(self.norm2(x),DEBUG=self.DEBUG)
        if DEBUG:
            print("  DEBUG CoreBlock:  after run       ", x.shape)                        
        x = x.transpose(-2, -1)
        if DEBUG:
            print("  DEBUG CoreBlock:  after transpose ", x.shape)                     
        return x

class CoreNet(nn.Module):
    def __init__(self, 
                 in_features=8, 
                 expansion=2,  
                 fn='dense',
                 kernel_size=1, 
                 act_layer='gelu', 
                 drop=0.,
                 DEBUG=False):
        super(CoreNet, self).__init__()
                    
        self.core1 = nn.Linear(in_features,   in_features * expansion  )
        self.core2 = nn.Linear(in_features * expansion , in_features )                                   
            
        if act_layer=='gelu':
            self.act = nn.GELU()
        elif act_layer=='relu':
            self.act = nn.ReLU()            
          
        self.drop = nn.Dropout(drop)
    def forward(self, x, DEBUG=False):
        if DEBUG:
            print("    DEBUG CoreNet x                ", x.shape)
        x = self.core1(x)
        if DEBUG:
            print("    DEBUG CoreNet x after core1    ", x.shape)        
        x = self.act(x)
        if DEBUG:
            print("    DEBUG CoreNet x after act      ", x.shape)        
        x = self.drop(x)
        if DEBUG:
            print("    DEBUG CoreNet x after drop     ", x.shape)        
        x = self.core2(x)
        if DEBUG:
            print("    DEBUG CoreNet x after core2    ", x.shape)        
        x = self.drop(x)
        if DEBUG:
            print("    DEBUG CoreNet x after drop     ", x.shape)        
        return x
     
