

import torch
import torchvision.transforms as tr
import torch.nn.functional as F
import torchsummaryX
import torch.nn as nn
import warnings

from mmcv.cnn import ConvModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import math

import os

from mmcv.utils import Registry

from ...utils.encoderdecoder import resize, _transform_inputs
from ...network_builder import DECODER



class RLN(nn.Module):
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)


    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


        
class FeatureMix_Decoder(nn.Module):
    def __init__(self, 
                 embed_dim = 512, 
                 norm_layer = nn.LayerNorm, 
                 use_norm = True):
        super(FeatureMix_Decoder, self).__init__()
        
        self.norm_layer_1 = norm_layer(embed_dim) if use_norm is True else nn.Identity()
        
        self.linear_1 = nn.Linear(embed_dim, embed_dim//2)
        
        self.conv_2 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2, kernel_size=3, padding=1)
        
        self.conv_3 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)
        self.act_layer1 = nn.GELU()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
                          
                          
    def forward(self, x):
        B, C, H, W = x.shape
    
        x = self.norm_layer_1(x.flatten(2).permute(0, 2, 1))
        identity = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        x1 = self.linear_1(x).permute(0, 2, 1).reshape(B, C//2, H, W)
        
        x2 = self.conv_2(x.permute(0, 2, 1).reshape(B, C, H, W))
        
        x = self.act_layer1(self.conv_3(torch.concat([x1, x2], dim=1))) + identity
        return x


class ConvMix_Decoder(nn.Module):
    def __init__(self, 
                 embed_dim = 512, 
                 use_norm = True):
        super(ConvMix_Decoder, self).__init__()
        
        self.norm_layer_1 = nn.LayerNorm(embed_dim) if use_norm is True else nn.Identity()
        self.norm_layer_2 = nn.LayerNorm(embed_dim) if use_norm is True else nn.Identity()
        
        self.linear_1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.conv_2 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        
        self.conv_3 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)
        self.act_layer_1 = nn.GELU()
          
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        
    def forward(self, x):       
        B, C, H, W = x.shape

        x = self.norm_layer_1(x.flatten(2).permute(0, 2, 1))
        identity = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        x_1 = self.linear_1(x).permute(0, 2, 1).reshape(B, C, H, W)
        
        x_2 = self.conv_2(x.permute(0, 2, 1).reshape(B, C, H, W))
        
        x = self.act_layer_1(self.conv_3(x_1 + x_2)) + identity
                      
        return x
        
        
class MixDecoderBlock(nn.Module):
    def __init__(self, feature_channels = 512, embed_dim = 512, norm_layer = nn.LayerNorm, use_norm = True):
        super(MixDecoderBlock, self).__init__()
        
        self.convmix_decoder =  ConvMix_Decoder(embed_dim=embed_dim,
                                                use_norm=use_norm)

        self.reduction_channels = nn.Conv2d(in_channels=embed_dim+feature_channels, 
                                            out_channels=embed_dim, 
                                            kernel_size=1
                                            )   
        
        self.feature_mix_decoder = FeatureMix_Decoder(embed_dim=embed_dim,
                                                     norm_layer=norm_layer,
                                                     use_norm=use_norm
                                                     )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
                  
    def forward(self, x, feature):
        x = self.convmix_decoder(x)
        
        x = torch.cat([x, feature], dim=1)
        
        x = self.reduction_channels(x)
        
        x = self.feature_mix_decoder(x)

        return x
        
        
@DECODER.register_module()
class Eh_Joint_ID_Head(nn.Module):

    def __init__(self, img_size, in_channels, embed_dim, num_classes, depths, use_norm=True, up_ratio=4, **kwargs):
        super(Eh_Joint_ID_Head, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.init_enhanced = nn.Sequential(nn.Conv2d(in_channels=c4_in_channels, out_channels=c4_in_channels, kernel_size=3, padding=1),
                                           nn.Conv2d(in_channels=c4_in_channels, out_channels=c4_in_channels, kernel_size=1)
                                           )
        
        self.init_act_layer = nn.GELU()
        
        # First option    
        self.c4_layer = nn.ModuleList([MixDecoderBlock(feature_channels=c4_in_channels,
                                                       embed_dim=c4_in_channels,
                                                       norm_layer=nn.LayerNorm,
                                                       use_norm=use_norm) for i in range(self.depths[3])])
        
        self.sigmoid_c4 = nn.Sigmoid()
        
        self.up_channels_c4 = nn.Conv2d(in_channels=c4_in_channels, out_channels=4*c3_in_channels, kernel_size=1)    
        self.up_sampling_c4 = nn.PixelShuffle(2)
        
        self.up_skip_conv_c4 = nn.Conv2d(in_channels=c4_in_channels, out_channels=up_ratio*c4_in_channels, kernel_size=3, padding=1)        
        self.up_sampling_skip_c4 = nn.PixelShuffle(8)       


        self.c3_layer = nn.ModuleList([MixDecoderBlock(feature_channels=c3_in_channels,
                                                       embed_dim=c3_in_channels,
                                                       norm_layer=nn.LayerNorm,
                                                       use_norm=use_norm) for i in range(self.depths[2])]) 

        self.sigmoid_c3 = nn.Sigmoid()
        
        self.up_channels_c3 = nn.Conv2d(in_channels=c3_in_channels, out_channels=4*c2_in_channels, kernel_size=1) 
        self.up_sampling_c3 = nn.PixelShuffle(2)
        
        self.up_skip_conv_c3 = nn.Conv2d(in_channels=c3_in_channels, out_channels=up_ratio*c3_in_channels, kernel_size=3, padding=1)  
        self.up_sampling_skip_c3 = nn.PixelShuffle(4)           
    
    
        self.c2_layer = nn.ModuleList([MixDecoderBlock(feature_channels=c2_in_channels,
                                                       embed_dim=c2_in_channels,
                                                       norm_layer=nn.LayerNorm,
                                                       use_norm=use_norm) for i in range(self.depths[1])]) 
        
        self.sigmoid_c2 = nn.Sigmoid()
        
        self.up_channels_c2 = nn.Conv2d(in_channels=c2_in_channels, out_channels=4*c1_in_channels, kernel_size=1) 
        self.up_sampling_c2 = nn.PixelShuffle(2)
        
        self.up_skip_conv_c2 = nn.Conv2d(in_channels=c2_in_channels, out_channels=up_ratio*c2_in_channels, kernel_size=3, padding=1)  
        self.up_sampling_skip_c2 = nn.PixelShuffle(2)  


        self.c1_layer = nn.ModuleList([MixDecoderBlock(feature_channels=c1_in_channels,
                                                       embed_dim=c1_in_channels,
                                                       norm_layer=nn.LayerNorm,
                                                       use_norm=use_norm) for i in range(self.depths[0])]) 
        
        self.sigmoid_c1 = nn.Sigmoid()
        
        self.up_skip_conv_c1 = nn.Conv2d(in_channels=c1_in_channels, out_channels=up_ratio*c1_in_channels, kernel_size=3, padding=1) 
        
        skip_in_channels = ((c4_in_channels//64) + (c3_in_channels//16) + (c2_in_channels//4) + c1_in_channels)*up_ratio
        self.up_skip_conv = nn.Conv2d(skip_in_channels, self.embed_dim, kernel_size=3, padding=1)        
        self.up_sampling_c1 = nn.PixelShuffle(4)
        
        in_channels = self.embed_dim // 16
        self.linear_pred1 = nn.Conv2d(in_channels, self.num_classes, kernel_size=3, padding=1)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, inputs):
        
        c1, c2, c3, c4 = inputs             # len=4, 1/4,1/8,1/16,1/32
        
        x = self.init_act_layer(self.init_enhanced(c4))
        
        for block in self.c4_layer:
            x = block(x, c4)
        x = self.sigmoid_c4(x)
        
        skip_c4 = self.up_skip_conv_c4(x)
        skip_c4 = self.up_sampling_skip_c4(skip_c4)

        x = self.up_channels_c4(x)
        x = self.up_sampling_c4(x)
                
        
        for block in self.c3_layer:
            x = block(x, c3)
        x = self.sigmoid_c3(x)
        
        skip_c3 = self.up_skip_conv_c3(x)
        skip_c3 = self.up_sampling_skip_c3(skip_c3)

        x = self.up_channels_c3(x)
        x = self.up_sampling_c3(x)


        for block in self.c2_layer:
            x = block(x, c2)
        x = self.sigmoid_c2(x)
        
        skip_c2 = self.up_skip_conv_c2(x)
        skip_c2 = self.up_sampling_skip_c2(skip_c2)

        x = self.up_channels_c2(x)
        x = self.up_sampling_c2(x)
         
         
        for block in self.c1_layer:
            x = block(x, c1)
        x = self.sigmoid_c1(x)
        
        skip_c1 = self.up_skip_conv_c1(x)
        skip_x = torch.cat([skip_c4, skip_c3, skip_c2, skip_c1], dim=1)
    
        x = self.up_skip_conv(skip_x)
        
        x = self.up_sampling_c1(x)
        x = self.linear_pred1(x)
        
        return x



@DECODER.register_module()
class joint_eh_single_head(Eh_Joint_ID_Head):
    def __init__(self, **kwargs):
        super(joint_eh_single_head, self).__init__(**kwargs,
                                                in_channels=[64, 128, 320, 512],
                                                depths=[3, 3, 3, 3],
                                                dropout_ratio=0.1,
                                                embed_dim=256,
                                                num_classes=3,
                                                use_norm=True,
                                                up_ratio=2
                                                )
