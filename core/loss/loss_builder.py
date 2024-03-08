

import torch
import torchvision.transforms as tr
import torch.functional as F
import torchsummaryX
import torch.nn as nn

from mmcv.utils import Registry


LOSS_BLOCK = Registry('loss_block')
LOSS_BUILDER = Registry('loss_builder')


@LOSS_BUILDER.register_module()
class Builder_Loss(nn.Module):
    def __init__(self, loss_build_list:list):
        super().__init__()
        
        builder_list = []
        for loss_build_cfg in loss_build_list:
            builder_list.append(LOSS_BUILDER.build(loss_build_cfg))
        
        self.builder_list = builder_list
        
        self.loss_tag_list = []
        for loss_builder in builder_list:
            for loss_tag in loss_builder.loss_tag_list:
                self.loss_tag_list.append(loss_tag)

    def forward(self, x, gt):
        final_loss = 0.0
        
        loss_final_list = []
        if isinstance(gt, list):
            for idx, loss_build in enumerate(self.builder_list):
                loss_result = loss_build.forward(x[idx], gt[idx])            
                final_loss += loss_result['final']
                
                for loss_value in loss_result['value_list']:
                    loss_final_list.append(loss_value)
        else:
            for loss_build in self.builder_list:
                loss_result = loss_build.forward(x, gt)            
                final_loss += loss_result['final']
                
                for loss_value in loss_result['value_list']:
                    loss_final_list.append(loss_value)
        
        loss = {'final': final_loss, 'value_list': loss_final_list}
        return loss


@LOSS_BUILDER.register_module()
class Build_DepthEstimation_Loss(nn.Module):
    def __init__(self, loss_cfg_list:list, depth_min_eval, total_loss_lamda=1.0):
        super().__init__()

        loss_list = []
        for loss_cfg in loss_cfg_list:
            loss_cfg['depth_min_eval'] = depth_min_eval
            loss_list.append(LOSS_BLOCK.build(loss_cfg))
        
        self.depth_estimation_loss = loss_list        
        self.total_loss_lamda = total_loss_lamda
        
        self.loss_tag_list =  self.create_loss_tag(loss_cfg_list)
        
    def forward(self, x, gt):
        final_loss = 0.0
        
        loss_value_list = []
        for criterion in self.depth_estimation_loss:
            loss_value = self.total_loss_lamda * criterion.forward(x, gt)
            final_loss += loss_value
            loss_value_list.append(loss_value)
        
        loss = {'final': final_loss, 'value_list': loss_value_list}
        return loss

    def create_loss_tag(self, loss_cfg_list):
        loss_tag_list = []
        for loss_cfg in loss_cfg_list:
            loss_tag_list.append(loss_cfg['type'])
        
        return loss_tag_list
    

@LOSS_BUILDER.register_module()
class Build_Enhancement_Loss(nn.Module):
    def __init__(self, loss_cfg_list:list, total_loss_lamda=1.0):
        super().__init__()

        loss_list = []
        for loss_cfg in loss_cfg_list:
            loss_list.append(LOSS_BLOCK.build(loss_cfg))
        
        self.enhancement_loss = loss_list  
        self.total_loss_lamda = total_loss_lamda      
        
        self.loss_tag_list =  self.create_loss_tag(loss_cfg_list)
        
    def forward(self, x, gt):
        final_loss = 0.0
        
        loss_value_list = []
        for criterion in self.enhancement_loss:
            loss_value = self.total_loss_lamda * criterion.forward(x, gt)
            final_loss += loss_value
            loss_value_list.append(loss_value)
        
        loss = {'final': final_loss, 'value_list': loss_value_list}
        return loss
    
    def create_loss_tag(self, loss_cfg_list):
        loss_tag_list = []
        for loss_cfg in loss_cfg_list:
            loss_tag_list.append(loss_cfg['type'])
        
        return loss_tag_list
    