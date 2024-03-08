
import torch.nn as nn
import torch
from ..loss_builder import LOSS_BLOCK


@LOSS_BLOCK.register_module()
class L1_loss(nn.Module):
    
    def __init__(self, lambda_l1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        
    def forward(self, generated, image_gt):
        torch_l1_dist = torch.nn.PairwiseDistance(p=1)
        loss = self.lambda_l1 * torch.mean(torch_l1_dist(generated, image_gt))
        
        return loss
    
    