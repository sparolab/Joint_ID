

import argparse
import os
from pathlib import Path

import torch
import sys


class MainParser():
    
    def __init__(self):

        FILE = Path(__file__).resolve()                     # 파일의 절대 경로
        ROOT = FILE.parents[1]                              # 파일의 상위의 상위 폴더
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))      # 워크스페이스에서의 상대 파일 경로 

        self.parser = argparse.ArgumentParser(description= "This is Initial Option for YGM_Monocular Camera Depth Estimation"
                                            , fromfile_prefix_chars='@')

        self.root = ROOT

        self.parser.convert_arg_line_to_args = self.convert_arg_line_to_args
        self.initialize()
        
    def initialize(self):
        
        # basic setting
        self.parser.add_argument('--project_name', type=str, default='test_U-gan', help='name of experiment')
        self.parser.add_argument('--mode', type=str, default='de_train', 
                                 help='train or test or use or sample_evaluate')        # train or test or use
        self.parser.add_argument('--encoder', type=str, default='densenet161_bts', 
                                                                    help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts')
        self.parser.add_argument('--device', type=str, required=True, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--hyp', type=str, help='hyperparmeters files for this project')
        self.parser.add_argument('--config', type=str, help='hyperparmeters files for this project')

        self.parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

        # Dataset
        self.parser.add_argument('--train_files', type=float, default=10, help='maximum depth in estimation') 
        self.parser.add_argument('--eval_files', type=str, default='', help='path to the filenames text file')
        self.parser.add_argument('--test_files', type=str, default='', help='path to the filenames text file')
        self.parser.add_argument('--input_height', type=int, default=480, help='input height')
        self.parser.add_argument('--input_width', type=int, default=640, help='input width')
        self.parser.add_argument('--data_path', type=str, required=False, help='path to the data for online evaluation')
        self.parser.add_argument('--gt_path', type=str, required=False, help='path to the groundtruth data for online evaluation')        
        self.parser.add_argument('--data_path_eval', type=str, required=False, help='path to the data for online evaluation')
        self.parser.add_argument('--gt_path_eval', type=str, required=False, help='path to the groundtruth data for online evaluation')

        # Log and save
        self.parser.add_argument('--model_save_directory', type=str, default= self.root / 'save/checkpoints', help='path to a checkpoint to load')
        self.parser.add_argument('--eval_log_directory', type=str, default='save/summaries/eval', help='output directory for eval summary,')
        self.parser.add_argument('--checkpoint_path', type=str, default=None, help='path to a checkpoint to load')
        self.parser.add_argument('--log_directory', type=str, default= self.root / 'save/summaries/log', help='model.pt와 훈련 tensorboard data를 저장할 폴더')
        self.parser.add_argument('--log_comment', type=str, default='', help='---')
        self.parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency in global steps')
        self.parser.add_argument('--save_freq', type=int, default=20, help='input width')
        
        # Logging Weights & Biases
        self.parser.add_argument('--wandb_restore_file_path', type=str, default='', help='path to a checkpoint to load')
        self.parser.add_argument('--wandb_save_path', type=str, default='dataset_root/wandb', help='path to a checkpoint to load')


        # Training
        self.parser.add_argument('--bts_size', type=int, default=512, help='initial num_filters in bts')
        self.parser.add_argument('--bn_no_track_stats', action='store_true', help='if set, will not track running stats in batch norm layers')
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        self.parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay factor for optimization')
        self.parser.add_argument('--adam_eps', type=float, default=1e-6, help='epsilon in Adam optimizer')
        self.parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
        self.parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--retrain', action='store_true', help='if used with checkpoint_path, will restart training from step zero')
        self.parser.add_argument('--variance_focus', type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')
        self.parser.add_argument('--end_learning_rate', type=float, default=-1, help='end learning rate')

        # Testing
        self.parser.add_argument('--save_lpg', action='store_true', help='chat it if you want to save every log files and images')


        # Preprocessing
        self.parser.add_argument('--use_right', action='store_true', help='if set, will randomly use right images when train on KITTI')
        self.parser.add_argument('--do_kb_crop', action='store_true', help='if set, crop input images as kitti benchmark images')
        self.parser.add_argument('--do_random_rotate', action='store_true', help='if set, will perform random rotation for augmentation')
        self.parser.add_argument('--degree', type=float, default=2.5, help='random rotation maximum degree')
        self.parser.add_argument('--do_random_crop', action='store_true', help='if set, crop randomly input images for argumentation')
        self.parser.add_argument('--do_random_flip', action='store_true', help='if set, flip randomly input images for argumentation')
        self.parser.add_argument('--do_augment_color', action='store_true', help='if set, changing randomly gamma, brightness, color augmentation of input images for argumentation')


        # Multi-gpu training
        self.parser.add_argument('--image_weights', action='store_true', help='use weighted image selection for training')
        self.parser.add_argument('--world_size', type=int, default=1, help='number of nodes for distributed training')        # world_size는 분산훈련에 쓸 노드의 개수
        self.parser.add_argument('--num_threads', type=int, default=1, help='number of threads to use for data loading')
        self.parser.add_argument('--gpu', type=int, default=None, help='GPU id to use.') 
        self.parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
        self.parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:1234', help='url used to set up distributed training')
        self.parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
        self.parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training to launch '
                                                                                            'N processes per node, which has N GPUs. This is the '
                                                                                            'fastest way to use PyTorch for either single node or '
                                                                                            'multi node data parallel training')

        # Online eval
        self.parser.add_argument('--do_online_eval', action='store_true', help='if set, perform online eval in every eval_freq steps')    # Tensorboard 이용하겠다는 뜻.
        self.parser.add_argument('--min_depth_eval', type=float, default=1e-3, help='minimum depth for evaluation')
        self.parser.add_argument('--max_depth_eval', type=float, default=80, help='maximum depth for evaluation')
        self.parser.add_argument('--eval_freq', type=int, default=500, help='Online evaluation frequency in global steps')
        self.parser.add_argument('--filenames_file_eval', type=str, default='', required=False, help='path to the filenames text file for online evaluation')



    def convert_arg_line_to_args(self, arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg


    def parse(self):
        
        if sys.argv.__len__() == 2:
            arg_filename_with_prefix = '@' + sys.argv[1]
            args = self.parser.parse_args([arg_filename_with_prefix])
        else:
            args = self.parser.parse_args()
            
        return args
