

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms as tr
from PIL import Image
import os
import random
import argparse
import numpy as np

import torchvision

from mmcv.utils import Registry
# from ..build___ import DATALOAD_BUILDER, DATALOADER

DATALOAD_BUILDER = Registry('dataload_builder')
DATALOADER = Registry('dataloader')


@DATALOAD_BUILDER.register_module()
class Build_DataLoader(object):
    def __init__(self, 
                 dataloader_dict, 
                 multiprocessing_distributed, 
                 mode: str,
                 batch_size = 1,
                 num_threads = 4
                 ):
        
        dataloader_dict['mode']=mode
        # dataloader_dict['transform']=self.preprocessing_transforms(mode)
        
        
        if mode == 'train':
            self.training_samples = DATALOADER.build(dataloader_dict)

            if multiprocessing_distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples, shuffle=False)
            else:
                self.train_sampler = None
            self.data = DataLoader(self.training_samples, 
                                    batch_size= batch_size,
                                    shuffle= False,
                                    num_workers= num_threads,
                                    pin_memory=True,
                                    drop_last=False,
                                    sampler= self.train_sampler
                                    )
        elif mode == 'eval':
            self.testing_samples = DATALOADER.build(dataloader_dict)
            
            if multiprocessing_distributed:
                self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 
                                    batch_size= 1,
                                    shuffle= False,
                                    num_workers= num_threads,
                                    pin_memory= True,
                                    sampler= self.eval_sampler
                                    )
            
        elif mode == 'test':
            self.testing_samples = DATALOADER.build(dataloader_dict)
            self.data = DataLoader(self.testing_samples, 
                                    batch_size= 1, 
                                    shuffle= False, 
                                    num_workers= 4
                                    )
        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

