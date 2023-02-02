# it's example code

it's WIP. please use it for reference code. do not share it in public. 

# source tree for MILK vocoder 

### model reated
```
-- model.py          : defime model 
   +-- config.json   : model parameters 
   +-- ffmixer.py    : feedforward-Mixer model 
   +-- fftr.py / transformer.py : transformer modules
   +-- rnn.py        : RNN modules 
      +-- lstm.py    : LSTM modules for RNN 
```

### train related 
```
-- run_slurm_1node_8gpu.sh  : example launch script with slurm environment 
-- train.py                 : train script ( define model, optimizer, loss, dataloader) with DDP and AMP
    +-- parser_config.py    : argment for train 
    +--utils.py             : utilities ( DDP, checkpoint, shutile)
    +-- lamp.py             : custom lamp optimizer 
   + --  train_epoch.py     : train script for each epoch (  loss , sample) 

```

### dataset related 
```
-- melgen_librosa.py   : dataloader with CPU version 
   + audioutil_librosa.py  : raw utility 
-- melgen_torch.py     : dataloader with CPU version 
   +-- audioutil_torch.py  : raw utility 
      +-- torch_stft.py    : low level module 
-- split_wav.py            : data preprocess 
-- get_filelists.py        : data preprocess  
        
```


- tip  1
[source copy at start stage ](https://github.com/yhgon/cau_nvidia/blob/main/code_examples/vocoder/train.py#L227) using [util script](https://github.com/yhgon/cau_nvidia/blob/main/code_examples/vocoder/utils.py#L67)

- tip 2. 
