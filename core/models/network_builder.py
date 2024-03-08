

import torch
import torchvision.transforms as tr
import torch.functional as F
import torchsummaryX
import torch.nn as nn
import warnings

from mmcv.cnn import ConvModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import math

import os
from mmcv.utils import Registry
# from ...build___ import MODEL_BUILDER, ENCODER, DECODER, TASK


MODEL_BUILDER = Registry('model_builder')
ENCODER  = Registry('encoder')
DECODER = Registry('decoder')
STRUCTURE = Registry('structure')
TASK = Registry('task')


@MODEL_BUILDER.register_module()
class Build_EncoderDecoder(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, img_size, backbone_pretrained_path=None, strict=True, task_cfg=None):
        super().__init__()
        
        encoder_cfg['img_size'] = self.init_img_size_check(img_size) 
        decoder_cfg['img_size'] = self.init_img_size_check(img_size) 
        
        self.encoder = ENCODER.build(encoder_cfg)
        self.decoder = DECODER.build(decoder_cfg)
        
        if task_cfg is not None:
            self.task = TASK.build(task_cfg)
        else:
            self.task = None

        if backbone_pretrained_path is not None:
            self.backbone_checkpoint_loader(checkpoint_path=backbone_pretrained_path, strict=strict)
            
    def backbone_checkpoint_loader(self,                     
                                   checkpoint_path,
                                   strict=True,
                                   device=None
                                   ):
        
        if os.path.isfile(checkpoint_path):
            if device is None:
                checkpoint = torch.load(checkpoint_path)
            else: 
                loc = 'cuda:{}'.format(device)
                checkpoint = torch.load(checkpoint_path, map_location= loc)        
            
            if not isinstance(checkpoint, dict):
                raise RuntimeError(f'No state_dict found in checkpoint file {checkpoint_path}')
            
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

            self.encoder.load_state_dict(state_dict, strict=strict)
        else:
            raise RuntimeError(f'No state_dict found in checkpoint file {checkpoint_path}')
    
    
    def init_img_size_check(self, img_size):
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            pass
        else:
            raise TypeError("Args: type'img_size' is must be 'int' or 'tuple', but Got {}".format(type(img_size)))     
        return img_size
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        if self.task is not None:
            x = self.task(x)
        
        return x

@MODEL_BUILDER.register_module()
class Build_OtherModels(nn.Module):
    def __init__(self, structure_cfg, img_size=None, strict=True, checkpoint_path=None, device= None):
        super().__init__()
        
        # if img_size is not None:
        # structure_cfg['img_size'] = self.init_img_size_check(img_size) 
        self.structure = STRUCTURE.build(structure_cfg)    

        if checkpoint_path is not None:
            self.structure = self.checkpoint_loader(checkpoint_path, self.structure, strict)
    
    
    def init_img_size_check(self, img_size):
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            pass
        else:
            raise TypeError("args: type'img_size' is must be 'int' or 'tuple', but Got {}".format(type(img_size)))
             
        return img_size


    def checkpoint_loader(self, checkpoint_path, model, strict=True):
        space1 = "".rjust(5)
        
        if os.path.isfile(checkpoint_path):
            print(space1+"🚀 Start Loading checkpoint '{}'".format(checkpoint_path))
            
            checkpoint = torch.load(checkpoint_path)
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
                
            model.load_state_dict(state_dict, strict=strict)
            
            print(space1+"🚀 Loaded checkpoint '{}'".format(checkpoint_path))
        else:
            print(space1+"🚀 No checkpoint found at '{}'".format(checkpoint_path))
            raise ValueError("🚀 No checkpoint found at '{}'".format(checkpoint_path))
        
        return model
        
    def forward(self, x):
        x = self.structure(x)
                
        return x



@MODEL_BUILDER.register_module()
class Build_Structure(nn.Module):
    def __init__(self, structure_cfg, img_size=None, strict=True, checkpoint_path=None, device= None):
        super().__init__()
        
        # if img_size is not None:
        structure_cfg['img_size'] = self.init_img_size_check(img_size) 
        self.structure = STRUCTURE.build(structure_cfg)    

        if checkpoint_path is not None:
            self.structure = self.checkpoint_loader(checkpoint_path, self.structure, strict)
    
    
    def init_img_size_check(self, img_size):
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            pass
        else:
            raise TypeError("args: type'img_size' is must be 'int' or 'tuple', but Got {}".format(type(img_size)))
             
        return img_size


    def checkpoint_loader(self, checkpoint_path, model, strict=True):
        space1 = "".rjust(5)
        
        if os.path.isfile(checkpoint_path):
            print(space1+"🚀 Start Loading checkpoint '{}'".format(checkpoint_path))
            
            checkpoint = torch.load(checkpoint_path)
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
                
            model.load_state_dict(state_dict, strict=strict)
            
            print(space1+"🚀 Loaded checkpoint '{}'".format(checkpoint_path))
        else:
            print(space1+"🚀 No checkpoint found at '{}'".format(checkpoint_path))
            raise ValueError("🚀 No checkpoint found at '{}'".format(checkpoint_path))
        
        return model
        
    def forward(self, x):
        x = self.structure(x)
                
        return x
    