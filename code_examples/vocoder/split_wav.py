import os 
import time
import argparse 
import multiprocessing
from multiprocessing import Pool 
from functools import partial
from joblib import Parallel


def save_wav(filepath, wav, sr=22050, subtype='PCM_16'):
    import soundfile as sf 
    sf.write(filepath, wav, sr, subtype=subtype)
    
def split_audio_batch(args, DEBUG=True):    
    import os
    import glob
    import time 
    
    work_dir = args.work_dir
    wav_dir  = args.wav_dir
    split_dir = args.split_dir
    hop_size  = args.hop_size
    num_frames = args.num_frames 
    max_splits = args.max_splits
    sr         = args.sampling_rate
    max_files   = args.max_files
    max_splits  = args.max_splits
    
    
    
    org_dir = os.path.join(work_dir, wav_dir)
    tar_dir = os.path.join(work_dir, split_dir)
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)    
    filenametxt = os.path.join(org_dir, 'LJ???-????.wav')
    filelists = sorted( glob.glob(filenametxt )  )
    if DEBUG :
        print("DEBUG : audio files {:d} files in {} to {} ".format(len(filelists), org_dir , tar_dir) )    

    tic = time.time()
    with Pool(args.num_cores) as pool :
        func = partial(split_audio, args=args)
        result = pool.map(func, filelists[:max_files])
    toc = time.time()
    dur = toc - tic 
    if DEBUG :
        print("DEBUG: it takes {:3.1f}min with {:d} files with {:d} cores ".format(dur/60, len(filelists),  args.num_cores ) )
    
def split_audio( filename, args,   DEBUG=False): 
    # 43 : 0.5sec, 86:1sec 
    import os
    import librosa
    import time

    
    work_dir = args.work_dir
    wav_dir  = args.wav_dir
    split_dir = args.split_dir
    hop_size  = args.hop_size
    num_frames = args.num_frames 
    max_splits = args.max_splits
    sr         = args.sampling_rate
    max_files   = args.max_files
    max_splits  = args.max_splits
    
    filename  = os.path.basename(filename) # if include directory
    org_dir = os.path.join(work_dir, wav_dir)
    tar_dir = os.path.join(work_dir, split_dir)
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
        
    input_filename = os.path.join(org_dir, filename)
    filebody       = os.path.splitext(filename)[0]
    ext            = os.path.splitext(filename)[1]
    
    wav_data, sr = librosa.core.load(input_filename, sr=sr, mono=True, res_type='polyphase')
    num_samples = len(wav_data)
    num_splits  = int(num_samples/(hop_size*num_frames))
    split_win = num_frames*hop_size 
    split_hop = split_win 
    #print(num_samples,num_splits, split_win, split_hop, wav_data  )
    
    tic = time.time()    
    for i in range(num_splits):
        if i < max_splits : 
            startpoint = i * split_hop
            endpoint   = startpoint + split_win
            file_split_name = os.path.join(tar_dir, filebody +'-{:03d}'.format(i)+ ext)
            wav_data_split = wav_data[startpoint :endpoint]
            #print(i, startpoint, endpoint, len(wav_data_split) , file_split_name)
            save_wav(file_split_name, wav_data_split, sr=sr, subtype='PCM_16' )
    toc = time.time()
    dur = toc -tic
    if DEBUG:
        print("it takes {:2.2f}sec for {:2.1f}sec audio with {:d}splits".format(dur, num_samples/sr, num_splits) )
    return dur 
    
    
if __name__ == '__main__':
    import argparse
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--work-dir',      type=str,  default='/scratch/hifigan/LJSpeech-1.1-raw/', help='work dlir')
    parser.add_argument('--wav-dir',       type=str,  default='wavs', help='wav directory')
    parser.add_argument('--split-dir',     type=str,  default='wavs_splits', help='split directory')
    parser.add_argument('--sampling-rate', type=int,  default=22050, help='sampling rate ')
    parser.add_argument('--hop-size',      type=int,  default=256 ,  help='hop_size')
    parser.add_argument('--num-frames',    type=int,  default=86,    help='frames 43 for 0.5sec 86 for 1sec')    
    parser.add_argument('--max-files',     type=int,  default=10,    help='handle max files for debug')
    parser.add_argument('--max-splits',    type=int,  default=4,     help='handle max splits for debug')    
    parser.add_argument('--num-cores',     type=int,  default=20,    help=' num of cores tos use ')       
    args = parser.parse_args()
    
    print(args)
    tic = time.time()
    print("multiproc start")
    split_audio_batch(args, DEBUG=True)
    toc = time.time()
    dur = toc -tic
    print("time", dur)   
    
