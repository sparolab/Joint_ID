
from local_configs.run_parser import MainParser
import yaml 

import argparse

import os

import torch.multiprocessing as mp
import torch

from mmcv.utils import Registry, Config, DictAction
from joint_train import Joint_Model_Train
from joint_test import Joint_Model_Test
from joint_test_samples import Joint_Model_Samples_Test
from joint_video_test import Joint_Model_Video_Test


        
def run_hyperparm_setting(opt, config):
    space1 = "".rjust(5)
    space2 = "".rjust(10)

    print("🚀 HyperParameters")
    for k, v in config.items():
        if isinstance(v, dict):
            print(space1 + f"{k}:")
            for k2, v2 in v.items():
                print(space2 + f"{k2}".ljust(20) + f"{v2}")
                opt.__dict__[k2] = v2
        else:
            opt.__dict__[k] = v
            print(space1 + f"{k}:".ljust(25) + f"{v}")

    return opt


def main(opt: argparse.Namespace):
    
    print("\n🚀🚀🚀 About the Parameters on this Project! 🚀🚀🚀")
    cfg = Config.fromfile(opt.config)

    opt = run_hyperparm_setting(opt, cfg)
    opt.log_comment = opt.config[:-3].split('/')[-1]
     
    if opt.mode == 'joint_train':
        Joint_Model_Train(opt).train()
        
    elif opt.mode == 'joint_test':
        Joint_Model_Test(opt).test()

    elif opt.mode == 'joint_samples_test':
        Joint_Model_Samples_Test(opt).samples_test()
        

        
    
if __name__ == '__main__':    
    parser = MainParser()
    
    opt = parser.parse()
    main(opt)
