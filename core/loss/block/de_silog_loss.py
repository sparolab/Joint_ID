
import torch.nn as nn
import torch
from ..loss_builder import LOSS_BLOCK


@LOSS_BLOCK.register_module()
class Silog_loss(nn.Module):
    def __init__(self, alpha_image_loss, depth_min_eval):
        self.alpha_image_loss = alpha_image_loss
        self.depth_min_eval = depth_min_eval

    def forward(self, depth_est, depth_gt):
        mask = depth_gt > self.depth_min_eval
        mask = mask.to(torch.bool)
        
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.alpha_image_loss * (d.mean() ** 2)) * 10.0
    
    
    