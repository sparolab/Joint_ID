
import torch.nn as nn
import torch
from ..loss_builder import LOSS_BLOCK


@LOSS_BLOCK.register_module()
class IGDL_Loss(nn.Module):
    
    def __init__(self, lambda_igdl, l1_or_l2 = 1):
        super().__init__()
        self.lambda_igdl = lambda_igdl
        self.l1_or_l2 = l1_or_l2
        
    def forward(self, generated, image_gt):
        image_gt_gradient_x = self.calculate_x_gradient(image_gt)
        generated_gradient_x = self.calculate_x_gradient(generated)
        image_gt_gradient_y = self.calculate_y_gradient(image_gt)
        generated_gradient_y = self.calculate_y_gradient(generated)
        pairwise_p_distance = torch.nn.PairwiseDistance(p=self.l1_or_l2)
        distances_x_gradient = pairwise_p_distance(
            image_gt_gradient_x, generated_gradient_x
        )
        distances_y_gradient = pairwise_p_distance(
            image_gt_gradient_y, generated_gradient_y
        )
        loss_x_gradient = torch.mean(distances_x_gradient)
        loss_y_gradient = torch.mean(distances_y_gradient)
        loss = 0.5 * (loss_x_gradient + loss_y_gradient)
        return loss * self.lambda_igdl
        
        
    def calculate_x_gradient(self, images):
        x_gradient_filter = torch.Tensor(
            [   [[0, 0, 0], 
                 [-1, 0, 1], 
                 [0, 0, 0]],
             
                [[0, 0, 0], 
                 [-1, 0, 1], 
                 [0, 0, 0]],
                
                [[0, 0, 0], 
                 [-1, 0, 1], 
                 [0, 0, 0]],
            ]
        ).cuda(images.device)
        x_gradient_filter = x_gradient_filter.view(3, 1, 3, 3)
        result = torch.functional.F.conv2d(
            images, x_gradient_filter, groups=3, padding=(1, 1)
        )
        return result
    

    def calculate_y_gradient(self, images):
        y_gradient_filter = torch.Tensor(
            [   [[0, 1, 0], 
                 [0, 0, 0], 
                 [0, -1, 0]],
             
                [[0, 1, 0], 
                 [0, 0, 0], 
                 [0, -1, 0]],
                
                [[0, 1, 0], 
                 [0, 0, 0], 
                 [0, -1, 0]],
            ]
        ).cuda(images.device)
        y_gradient_filter = y_gradient_filter.view(3, 1, 3, 3)
        result = torch.functional.F.conv2d(
            images, y_gradient_filter, groups=3, padding=(1, 1)
        )
        return result
    