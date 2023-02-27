
import argparse
import os

def config_all(parser, add_help=False):
    """
    Parse commandline arguments.
    """
    distributed = parser.add_argument_group('distributed setup')

    parser.add_argument('-o', '--output-dir',  type=str, default='./output',  help='Directory to save checkpoints')  
    parser.add_argument(      '--log-file',    type=str, default='./output',  help='Path to a DLLogger log file')    
    parser.add_argument(      '--sample-dir',  type=str, default='./samples', help='sample files ')    
               
    distributed.add_argument('--local_rank',                type=int, default=os.getenv('LOCAL_RANK', 0),  help='Rank of the process for multiproc. Do not set manually.')
    distributed.add_argument('--world_size',                type=int, default=os.getenv('WORLD_SIZE', 1),  help='Number of processes for multiproc. Do not set manually.')    

    training = parser.add_argument_group('training setup')
    training.add_argument('-bs', '--batch-size',            type=int,   default=2,      help='Batch size per GPU')
    
    training.add_argument('--resume',                       action='store_true',        help='Resume training from the last available checkpoint')       
    training.add_argument('--cuda',                         action='store_true',        help='Run on GPU using CUDA')
    training.add_argument('--cudnn-benchmark',              action='store_true',        help='Enable cudnn benchmark mode')    
    training.add_argument('--fp16',                         action='store_true',        help='Run training in fp16/mixed precision')
    
    training.add_argument('--epochs',                       type=int, default=10,        help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint',        type=int, default=1,        help='Number of epochs per checkpoint')    
    training.add_argument('--print-per-iter',               type=int, default=1,       help='Number of iter  per print')    
    training.add_argument('--seed',                         type=int, default=1234,     help='Seed for PyTorch random number generators')     
    training.add_argument('--amp',                          type=str,   default='pytorch', choices=['apex', 'pytorch'],      help='Implementation of automatic mixed precision')    
    training.add_argument('--apex_amp_opt_level',           type=str,   default='O2',   choices=['O0', 'O1', 'O2', 'O3'], help='Optimization level for apex amp')
    training.add_argument('--multi_gpu',                    type=str,   default='ddp',  choices=['ddp', 'dp'],            help='Use multiple GPU')
    training.add_argument('--ema-decay',                    type=float, default=0,      help='Discounting factor for training weights EMA')
    training.add_argument('--gradient-accumulation-steps',  type=int,   default=1,      help='Training steps to accumulate gradients for')
    training.add_argument('--num_samples',                  type=int,   default=1,      help='limit the number of samples to generate')
    
    data = parser.add_argument_group('data setup')
    data.add_argument('--loadfromfile',  action='store_true',        help='load mel from disk')       
    data.add_argument('--meltofile',     action='store_true',        help='save mel to disk')    
    data.add_argument('--cache',         action='store_true',        help='use cache mode')     
    data.add_argument('--shuffle',       action='store_true',        help='shuffle option for dataset')         
 
    data.add_argument('--spec_option',   type=str,   default=7,      choices=[1,2,3,4,5,6,7],      help='Implementation of automatic mixed precision')       
     
    data.add_argument('--n_cache_reuse', type=int, default=1,                help='cache mode option')  
    data.add_argument('--dataset-dir',   type=str, default='/dataset/LJSpeech-1.1', help='Path to dataset')
    data.add_argument('--wavs-dir',      type=str, default='wavs',           help='filelist for test ')    
    data.add_argument('--mels-dir',      type=str, default='mels',           help='filelist for test ')    
    data.add_argument('--ext_mel_file',  type=str, default='.pt',            help='mel file ext name')      
    
    data.add_argument('--filelists-dir',   type=str, default='./',            help='Path to filelist')    
    data.add_argument('--filelists-train', type=str, default='filelist_train.txt',     help='filelist for train')
    data.add_argument('--filelists-valid', type=str, default='filelist_valid.txt',     help='filelist for valid')
    data.add_argument('--filelists-test',  type=str, default='filelist_test.txt',      help='filelist for test ')       
    
    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optim',                             type=str,   default='lamb',    choices=['adam', 'sgd', 'adagrad', 'lamb', 'jitlamb'],  help='Optimizer to use')    
    opt.add_argument(       '--lr-schedulers',              type=str,   default='invsqrt', choices=['invsqrt' ],  help='learning rate scheduler')        
    opt.add_argument('-lr', '--learning-rate',              type=float, default=0.1,      help='Learing rate')
    opt.add_argument(       '--weight-decay',               type=float, default=1e-6,     help='Weight decay')
    opt.add_argument(       '--clip',                       type=float, default=0.25,     help='Clip threshold for gradients')    
    opt.add_argument(       '--grad-clip-thresh',           type=float, default=1000.0,   help='Clip threshold for gradients')
    opt.add_argument(       '--warmup-steps',               type=int,   default=1000,     help='Number of steps for lr warmup')
                                         
    return parser
