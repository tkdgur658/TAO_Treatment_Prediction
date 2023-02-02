
import torch
import torch.nn as nn
import torch.functional as F  
    
class Fake_Model(nn.Module):
    def __init__(self, args, args_data, args_model):
        super(Fake_Model, self).__init__()
        self.norm_mel   = args_data.norm_mel
        self.norm_mag   = args_data.norm_mag
        self.norm_lossy = args_data.norm_lossy
        self.norm_rabs  = args_data.norm_rabs
        self.norm_rzc   = args_data.norm_rzc
        self.DEBUG      = args.DEBUG 
        self.nframes    = int(args_data.max_nframes + 1 )
                
        self.linear  = nn.Linear( args_model.nse_dmodel,  args_model.nse_output_channels, bias=False)                    

    def forward(self, inputs, DEBUG=False):
        out = inputs[:,0,0,0,0]
        return out 
    
    def infer(self, inputs, DEBUG=False):    
        return self.forward(inputs, DEBUG=DEBUG)
   
    
class Fake_Model_1(nn.Module):
    def __init__(self, args, args_data, args_model):
        super(Fake_Model, self).__init__()
        self.norm_mel   = args_data.norm_mel
        self.norm_mag   = args_data.norm_mag
        self.norm_lossy = args_data.norm_lossy
        self.norm_rabs  = args_data.norm_rabs
        self.norm_rzc   = args_data.norm_rzc
        self.DEBUG      = args.DEBUG 
        self.nframes    = int(args_data.max_nframes + 1 )
        
        self.proj_pre   = nn.Linear( args_model.nse_input_channels,  args_model.nse_dmodel, bias=False) 
        self.proj_post  = nn.Linear( args_model.nse_dmodel,  args_model.nse_output_channels, bias=False)                    

    def forward(self, inputs, DEBUG=False):
        # inputs    : nframes, input_data(batch, nframes, nbins)
        # outputs   : nframes, output_data(batch, nframes, nbins)
        nframes, out = inputs
        out = self.proj_pre(out)
        out = self.proj_post(out)                                     
        return out 
    
    def infer(self, inputs, DEBUG=False):    
        return self.forward(inputs, DEBUG=DEBUG)