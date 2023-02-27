
import argparse
import os

def config_all(parser, add_help=False):
    parser.add_argument('-et', '--Experiments-Time',        type=str, default='000000_000000', help='Experiments Start Time')  
    parser.add_argument('--seed',                           type=int, default=0, help='Random Seed')
    parser.add_argument('-mun', '--module-name',            type=str, default='ResNet_3D', help='Module Name List of Model')
    parser.add_argument('-men', '--model-name',             type=str, default='ResNet18_3D', help='Model Name List')
    parser.add_argument('-medir', '--model-dir',            type=str, default='models', help='Model Directory')
    parser.add_argument('-nc', '--num-classes',             type=int, default=1, help='Number of Classes')
    parser.add_argument('-ic', '--in-channels',             type=int, default=1, help='Number of Input Channels')
    
    parser.add_argument('-dd', '--data-dir',                type=str, default='input_example', help='Numpy Dataset Dir')    
    parser.add_argument('-lf', '--label-file',              type=str, default='target_example.csv', help='Target CSV Dir')    
    parser.add_argument('-od', '--output-dir',              type=str, default='output', help='Output Directory')
    parser.add_argument('-dv', '--device',                  type=str, default='cuda', help='force CPU mode for debug')
    
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--local_rank',                type=int, default=os.getenv('LOCAL_RANK', 0),  help='Rank of the process for multiproc. Do not set manually.')
    distributed.add_argument('--world_size',                type=int, default=os.getenv('WORLD_SIZE', 1),  help='Number of processes for multiproc. Do not set manually.')    

    training = parser.add_argument_group('training setup')
    training.add_argument('-bs', '--Batch-Size',            type=int,   default=2,      help='Batch size per GPU')
    training.add_argument('-ep', '--Epochs',                type=int, default=10,        help='Number of total epochs to run')
    training.add_argument('--amp',                          type=str,   default='pytorch', choices=['apex', 'pytorch'],      help='Implementation of automatic mixed precision')
    training.add_argument('--apex_amp_opt_level',           type=str,   default='O2',   choices=['O0', 'O1', 'O2', 'O3'], help='Optimization level for apex amp')
    training.add_argument('--multi_gpu',                    type=str,   default='ddp',  choices=['ddp', 'dp'],            help='Use multiple GPU')
    
    training.add_argument('--resume',                       action='store_true',        help='Resume training from the last available checkpoint')       
    training.add_argument('--cuda',                         action='store_true',        help='Run on GPU using CUDA')
    training.add_argument('--cudnn-benchmark',              action='store_true',        help='Enable cudnn benchmark mode')    
    training.add_argument('--fp16',                         action='store_true',        help='Run training in fp16/mixed precision')
    
    
    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optim',                             type=str,   default='lamb',    choices=['adam', 'sgd', 'adagrad', 'lamb', 'jitlamb'],  help='Optimizer to use')    
    opt.add_argument(       '--lr-schedulers',              type=str,   default='invsqrt', choices=['invsqrt' ],  help='learning rate scheduler')        
    opt.add_argument('-lr', '--learning-rate',              type=float, default=0.1,      help='Learing rate')
    opt.add_argument(       '--weight-decay',               type=float, default=1e-4,     help='Weight decay')
    opt.add_argument(       '--clip',                       type=float, default=0.25,     help='Clip threshold for gradients')    
    opt.add_argument(       '--grad-clip-thresh',           type=float, default=1000.0,   help='Clip threshold for gradients')
    opt.add_argument(       '--warmup-steps',               type=int,   default=1000,     help='Number of steps for lr warmup')
    return parser
