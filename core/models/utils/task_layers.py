

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

from ..network_builder import TASK


@TASK.register_module()
class DepthEstimation_Task(nn.Module):
    def __init__(self, max_depth):
        super(DepthEstimation_Task, self).__init__()
        
        self.max_depth = max_depth
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(x)*self.max_depth
        return x


@TASK.register_module()
class Enhancement_Task(nn.Module):
    def __init__(self):
        super(Enhancement_Task, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(x)
        return x