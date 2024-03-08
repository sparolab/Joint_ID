
import torch.nn as nn
import torch
from ..loss_builder import LOSS_BLOCK


@LOSS_BLOCK.register_module()
class L2_loss(nn.Module):
    
    def __init__(self, lambda_l2):
        super().__init__()
        self.lambda_l2 = lambda_l2
        
    def forward(self, generated, image_gt):
        torch_l1_dist = torch.nn.PairwiseDistance(p=2)
        loss = self.lambda_l2 * torch.mean(torch_l1_dist(generated, image_gt))
        
        return loss
    
    