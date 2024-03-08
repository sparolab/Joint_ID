
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
import torchsummaryX
from PIL import Image
from ..loss_builder import LOSS_BLOCK


@LOSS_BLOCK.register_module()
class LaplacianLoss(nn.Module):
    def __init__(self, lambda_laplacian):
        super().__init__()
        
        self.lambda_laplacian = lambda_laplacian
        
        # 파이토치에서는 커널을 다음과 같이 정의해줘야한다. (channels, 1, size, size)
        laplacian_kernel = torch.Tensor([[[0,   -1,    0],
                                          [-1,   4,   -1],
                                          [0,   -1,    0]],
                                         
                                         [[0,   -1,    0],
                                          [-1,   4,   -1],
                                          [0,   -1,    0]],
                                         
                                         [[0,   -1,    0],
                                          [-1,   4,   -1],
                                          [0,   -1,    0]]], 
                                        )
        
        self.laplacian_kernel = laplacian_kernel.unsqueeze(1)
        self.laplacian_l2_norm = torch.nn.PairwiseDistance(p=2)
    
    def forward(self, inputs, targets):  
        
        laplacian_kernel = self.laplacian_kernel.to(inputs.device) 
        laplacian_l2_norm = self.laplacian_l2_norm.to(inputs.device)
        
        inputs_result = torch.functional.F.conv2d(inputs, laplacian_kernel, groups=3, padding=(1,1))
        targets_result = torch.functional.F.conv2d(targets, laplacian_kernel, groups=3, padding=(1,1))
        
        return self.lambda_laplacian * torch.mean(laplacian_l2_norm(inputs_result, targets_result))
        
