


import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import os

from timm.models.layers import to_2tuple
from collections import namedtuple
from ..network_builder import STRUCTURE, MODEL_BUILDER, Build_EncoderDecoder




class Joint_Block(nn.Module):
    def __init__(self, structure_in_channel, joint_in_channel):
        super().__init__()
        in_channel = structure_in_channel + joint_in_channel
        self.joint_layer1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
                                          nn.Conv2d(in_channels=in_channel, out_channels=structure_in_channel, kernel_size=1),
                                          ) 
        
        self.joint_layer2 = nn.Conv2d(in_channels=in_channel, out_channels=structure_in_channel, kernel_size=1) 
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_() 
       
    def forward(self, structure_x, joint_x):
        input_x = torch.concat([structure_x, joint_x], dim=1)
        x1 = self.joint_layer1(input_x)
        x2 = self.joint_layer2(input_x)
        x = x1 + x2
        
        return x

@STRUCTURE.register_module()
class Joint_ID(nn.Module):
    def __init__(self,
                 img_size, 
                 depth_model_cfg, 
                 enhanced_model_cfg, 
                 de_checkpoint, 
                 de_strict, 
                 eh_checkpoint, 
                 eh_strict,
                 is_de_no_grad,
                 is_eh_no_grad,
                 ):
        super(Joint_ID, self).__init__()

        depth_model_cfg['img_size'] = img_size         
        enhanced_model_cfg['img_size'] = img_size 
        
        self.depth_structure = Build_EncoderDecoder(**depth_model_cfg)
        self.enhanced_structure = Build_EncoderDecoder(**enhanced_model_cfg)
        
        if is_de_no_grad is True:
            for p in self.depth_structure.encoder.parameters():
                p.requires_grad=False
        
        if is_eh_no_grad is True:
            for p in self.enhanced_structure.encoder.parameters():
                p.requires_grad=False
        
        self.decoder_de_in_channels = self.depth_structure.decoder.in_channels
        depth_c1_in_channels, depth_c2_in_channels, depth_c3_in_channels, depth_c4_in_channels = self.decoder_de_in_channels
        
        self.decoder_eh_in_channels = self.enhanced_structure.decoder.in_channels
        enhanced_c1_in_channels, enhanced_c2_in_channels, enhanced_c3_in_channels, enhanced_c4_in_channels = self.decoder_eh_in_channels
        
        self.depth_embed_dim = self.depth_structure.decoder.embed_dim
        self.enhanced_embed_dim = self.enhanced_structure.decoder.embed_dim        
        
        self.decoder_de_joint_block_1 = Joint_Block(depth_c4_in_channels, enhanced_c4_in_channels)
        self.decoder_de_joint_block_2 = Joint_Block(depth_c3_in_channels, enhanced_c3_in_channels)
        self.decoder_de_joint_block_3 = Joint_Block(depth_c2_in_channels, enhanced_c2_in_channels)
        self.decoder_de_joint_block_4 = Joint_Block(depth_c1_in_channels, enhanced_c1_in_channels)
        
        self.decoder_eh_joint_block_1 = Joint_Block(enhanced_c4_in_channels, depth_c4_in_channels)
        self.decoder_eh_joint_block_2 = Joint_Block(enhanced_c3_in_channels, depth_c3_in_channels)
        self.decoder_eh_joint_block_3 = Joint_Block(enhanced_c2_in_channels, depth_c2_in_channels)
        self.decoder_eh_joint_block_4 = Joint_Block(enhanced_c1_in_channels, depth_c1_in_channels)

        if de_checkpoint is not None:
            self.depth_structure = self.checkpoint_loader(de_checkpoint, self.depth_structure, strict=de_strict)
        if eh_checkpoint is not None:
            self.enhanced_structure = self.checkpoint_loader(eh_checkpoint, self.enhanced_structure, strict=eh_strict)

        
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
            model.load_state_dict(checkpoint['model'], strict=strict)
            
            print(space1+"🚀 Loaded checkpoint '{}'".format(checkpoint_path))
        else:
            print(space1+"🚀 No checkpoint found at '{}'".format(checkpoint_path))
            raise ValueError("🚀 No checkpoint found at '{}'".format(checkpoint_path))

        return model
    
    def forward(self, x):
        
        
        ########################### Encoder part ############################# 
        depth_x = self.depth_structure.encoder(x)
        enhanced_x = self.enhanced_structure.encoder(x)

        depth_c1, depth_c2, depth_c3, depth_c4 = depth_x                            # len=4, 1/4,1/8,1/16,1/32
        enhanced_c1, enhanced_c2, enhanced_c3, enhanced_c4 = enhanced_x             # len=4, 1/4,1/8,1/16,1/32



        ########################### Decoder part ############################# 
        ######## for feature map 1/32 ########
        depth_x = self.depth_structure.decoder.init_depth(depth_c4)
        depth_x = self.depth_structure.decoder.init_act_layer(depth_x)
        
        enhanced_x = self.enhanced_structure.decoder.init_enhanced(enhanced_c4)
        enhanced_x = self.enhanced_structure.decoder.init_act_layer(enhanced_x)
        
            
        
        ######### for feature map 1/16 ######### 
        # 1/16 for depth 
        for block in self.depth_structure.decoder.c4_layer:
            depth_x = block(depth_x, depth_c4)
        depth_x_ = self.depth_structure.decoder.sigmoid_c4(depth_x)
              
        # 1/16 for enhanced 
        for block in self.enhanced_structure.decoder.c4_layer:
            enhanced_x = block(enhanced_x, enhanced_c4)
        enhanced_x_ = self.enhanced_structure.decoder.sigmoid_c4(enhanced_x)
        
        # 1/16 for joint depth
        depth_x = self.decoder_de_joint_block_1(depth_x_, enhanced_x_)
        skip_depth_c4 = self.depth_structure.decoder.up_skip_conv_c4(depth_x)
        skip_depth_c4 = self.depth_structure.decoder.up_sampling_skip_c4(skip_depth_c4)
        depth_x = self.depth_structure.decoder.up_channels_c4(depth_x)
        depth_x = self.depth_structure.decoder.up_sampling_c4(depth_x)

        # 1/16 for joint enhanced
        enhanced_x = self.decoder_eh_joint_block_1(enhanced_x_, depth_x_)
        skip_enhanced_c4 = self.enhanced_structure.decoder.up_skip_conv_c4(enhanced_x)
        skip_enhanced_c4 = self.enhanced_structure.decoder.up_sampling_skip_c4(skip_enhanced_c4)        
        enhanced_x = self.enhanced_structure.decoder.up_channels_c4(enhanced_x)
        enhanced_x = self.enhanced_structure.decoder.up_sampling_c4(enhanced_x)

        

        ######### for feature map 1/8 #########
        # 1/8 for depth 
        for block in self.depth_structure.decoder.c3_layer:
            depth_x = block(depth_x, depth_c3)
        depth_x_ = self.depth_structure.decoder.sigmoid_c3(depth_x)
        
        # 1/8 for enhanced 
        for block in self.enhanced_structure.decoder.c3_layer:
            enhanced_x = block(enhanced_x, enhanced_c3)
        enhanced_x_ = self.enhanced_structure.decoder.sigmoid_c3(enhanced_x)
        
        # 1/8 for joint depth
        depth_x = self.decoder_de_joint_block_2(depth_x_, enhanced_x_)
        skip_depth_c3 = self.depth_structure.decoder.up_skip_conv_c3(depth_x)
        skip_depth_c3 = self.depth_structure.decoder.up_sampling_skip_c3(skip_depth_c3)
        depth_x = self.depth_structure.decoder.up_channels_c3(depth_x)
        depth_x = self.depth_structure.decoder.up_sampling_c3(depth_x)        
        
        # 1/8 for joint enhanced
        enhanced_x = self.decoder_eh_joint_block_2(enhanced_x_, depth_x_)
        skip_enhanced_c3 = self.enhanced_structure.decoder.up_skip_conv_c3(enhanced_x)
        skip_enhanced_c3 = self.enhanced_structure.decoder.up_sampling_skip_c3(skip_enhanced_c3)        
        enhanced_x = self.enhanced_structure.decoder.up_channels_c3(enhanced_x)
        enhanced_x = self.enhanced_structure.decoder.up_sampling_c3(enhanced_x)        
    


        ######### for feature map 1/4 #########
        # 1/4 for depth 
        for block in self.depth_structure.decoder.c2_layer:
            depth_x = block(depth_x, depth_c2)
        depth_x_ = self.depth_structure.decoder.sigmoid_c2(depth_x)
        
        # 1/4 for enhanced 
        for block in self.enhanced_structure.decoder.c2_layer:
            enhanced_x = block(enhanced_x, enhanced_c2)
        enhanced_x_ = self.enhanced_structure.decoder.sigmoid_c2(enhanced_x)

        # 1/4 for joint depth
        depth_x = self.decoder_de_joint_block_3(depth_x_, enhanced_x_)
        skip_depth_c2 = self.depth_structure.decoder.up_skip_conv_c2(depth_x)
        skip_depth_c2 = self.depth_structure.decoder.up_sampling_skip_c2(skip_depth_c2)
        depth_x = self.depth_structure.decoder.up_channels_c2(depth_x)
        depth_x = self.depth_structure.decoder.up_sampling_c2(depth_x)               
        
        # 1/4 for joint enhanced
        enhanced_x = self.decoder_eh_joint_block_3(enhanced_x_, depth_x_)      
        skip_enhanced_c2 = self.enhanced_structure.decoder.up_skip_conv_c2(enhanced_x)
        skip_enhanced_c2 = self.enhanced_structure.decoder.up_sampling_skip_c2(skip_enhanced_c2)        
        enhanced_x = self.enhanced_structure.decoder.up_channels_c2(enhanced_x)
        enhanced_x = self.enhanced_structure.decoder.up_sampling_c2(enhanced_x)           
        
          
        
        ######### for feature map original size ######### 
        # original size for depth 
        for block in self.depth_structure.decoder.c1_layer:
            depth_x = block(depth_x, depth_c1)
        depth_x_ = self.depth_structure.decoder.sigmoid_c1(depth_x)
        
        # original size for enhanced 
        for block in self.enhanced_structure.decoder.c1_layer:
            enhanced_x = block(enhanced_x, enhanced_c1)
        enhanced_x_ = self.enhanced_structure.decoder.sigmoid_c1(enhanced_x)
        
        # original size for joint depth
        depth_x = self.decoder_de_joint_block_4(depth_x_, enhanced_x_)
        skip_depth_c1 = self.depth_structure.decoder.up_skip_conv_c1(depth_x)
                
        # original size for joint enhanced
        enhanced_x = self.decoder_eh_joint_block_4(enhanced_x_, depth_x_)
        skip_enhanced_c1 = self.enhanced_structure.decoder.up_skip_conv_c1(enhanced_x)
        
        
        
        ######### for feature map final #########
        # create final depth img
        skip_depth_x = torch.cat([skip_depth_c4, skip_depth_c3, skip_depth_c2, skip_depth_c1], dim=1)
        depth_x = self.depth_structure.decoder.up_skip_conv(skip_depth_x)
        depth_x = self.depth_structure.decoder.up_sampling_c1(depth_x)
        depth_x = self.depth_structure.decoder.linear_pred1(depth_x)             
        
        if self.depth_structure.task is not None:
            depth_x = self.depth_structure.task(depth_x)

        # create final enhanced img
        skip_enhanced_x = torch.cat([skip_enhanced_c4, skip_enhanced_c3, skip_enhanced_c2, skip_enhanced_c1], dim=1)
        enhanced_x = self.enhanced_structure.decoder.up_skip_conv(skip_enhanced_x)
        enhanced_x = self.enhanced_structure.decoder.up_sampling_c1(enhanced_x)
        enhanced_x = self.enhanced_structure.decoder.linear_pred1(enhanced_x)        

        if self.enhanced_structure.task is not None:
            enhanced_x = self.enhanced_structure.task(enhanced_x)        
                
        return depth_x, enhanced_x
        

