
def filelists(work_dir='/scratch/hifigan/LJSpeech-1.1-raw/', wav_dir='wavs', split_dir='wavs_splits', output_filename='splits_lists.txt', DEBUG=False): 
    import glob
    import os 
    import time 
    org_dir = os.path.join(work_dir, wav_dir)
    tgt_dir = os.path.join(work_dir, split_dir)
    org_lists = sorted(glob.glob( org_dir + '/*.wav' ) )
    split_lists = sorted(glob.glob( tgt_dir + '/*.wav' ) )
    with open(output_filename, 'w') as outfile :
        for i, file in enumerate(split_lists):
            filename = os.path.basename(file)
            filebody = os.path.splitext(filename)[0]
            outfile.write("{}\n".format(filebody ) )
    if DEBUG : 
        print("list up {} files  to {} ".format(len(split_lists ), output_filename ) )

def rawcount(filename):
    f = open(filename, 'rb')
    lines =0
    buf_size = 1024*1024
    read_f = f.raw.read
    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)
    return lines 
def slicefile(filename, start, end):
    import itertools
    lines = open(filename)
    return itertools.islice(lines, start, end)

def writeline(file, target, start, end):
    out = open(target,'w')
    for line in slicefile(file, start, end):
        out.write(line)
        
def split_train_val(all_filename, test_filename  = 'test.txt',  train_filename = 'train.txt', valid_filename = 'valid.txt',  test_len=2000, val_len =400):      
    
    total_lines   = rawcount(all_filename)
 
    test_start    = 0
    test_end      = test_len
    valid_start   = test_len
    valid_end     = test_len + val_len
    train_start   = test_len + val_len
    train_end     = total_lines
    print("spilt all:{} files to test:{} valid:{} train:{} ".format(total_lines, test_end-test_start, valid_end-valid_start, train_end-train_start ))

    writeline(all_filename, train_filename, train_start, train_end)
    writeline(all_filename, valid_filename, valid_start, valid_end)
    writeline(all_filename, test_filename,  test_start,  test_end)    
    
if __name__ == '__main__':
    import argparse
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--work-dir',        type=str, default='/scratch/hifigan/LJSpeech-1.1-raw/', help='work dlir')
    parser.add_argument('--wav-dir',         type=str, default='wavs', help='wav directory')
    parser.add_argument('--split-dir',       type=str, default='wavs_splits', help='split directory')
    parser.add_argument('--all_filename', type=str, default='filelist_all_splits.txt', help='sampling rate ')
    parser.add_argument('--train-filename',  type=str, default='filelist_train.txt', help='hop_size')
    parser.add_argument('--test-filename',   type=str, default='filelist_test.txt',  help='frames 43 for 0.5sec 86 for 1sec')    
    parser.add_argument('--valid-filename',  type=str, default='filelist_valid.txt',  help='handle max files for debug')
    parser.add_argument('--test-len',      type=int,  default=2000,    help='handle max splits for debug')    
    parser.add_argument('--val-len',       type=int,  default=400,   help=' num of cores tos use ')       
    args = parser.parse_args()
    
    print(args)
    filelists(work_dir=args.work_dir, wav_dir=args.wav_dir, split_dir=args.split_dir, output_filename=args.all_filename, DEBUG=False)
    split_train_val(all_filename=args.all_filename, test_filename  = args.test_filename, train_filename = args.train_filename, valid_filename =args.valid_filename,  test_len=args.test_len, val_len =args.val_len)


    
