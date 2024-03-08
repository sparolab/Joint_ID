

import cv2
import torch

import numpy as np

from PIL import Image
from torchvision import transforms


class AttentionMapVisualizing:
    def __init__(self, 
                 head_fusion='mean',
                 discard_ratio=0.9):
        
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
    
    def __call__(self, attention_list):
        result_mask_list = []
        
        for attention in attention_list:
            try:
                batch, channels, height, width = attention.size()
            except:
                batch, channels, height, width = attention.shape
            
            result = torch.eye(n=width, m=height)
            
            if self.head_fusion == 'mean':
                attention_fused = attention.mean(axis=1)
            
            elif self.head_fusion == 'max':
                attention_fused = attention.max(axis=1)[0]
            
            elif self.head_fusion == 'min':
                attention_fused = attention.min(axis=1)[0]
            
            else:
                raise ValueError("GM: Attention head Fusion type not supported")
               
            attention_fused = attention_fused / attention_fused.max()
            # print('attention_fused.shape: ', attention_fused.shape)
            result_mask_list.append(attention_fused)
        
        return result_mask_list
            
    def show_mask_on_image(self, img, mask, color=cv2.COLORMAP_JET):
        mask = mask / (mask.max() - mask.min())
        mask = np.uint8(255*mask)
        print('mask.type: ', type(mask))
        heatmap = cv2.applyColorMap(mask, color)
        heatmap = np.float32(heatmap) / 255.0
        
        cam = heatmap*2.0 + np.float32(img*1.0)
        cam = cam / np.max(cam)
        
        return np.uint8(255 * cam)
    
    def show_mask(self, mask, color=cv2.COLORMAP_JET):
        # mask = mask / (mask.max() - mask.min())
        mask = (mask - mask.min()) / (mask.max() - mask.min())

        try:
            mask = np.uint8(255*mask).transpose(1,2,0)
        except:
            mask = np.uint8(255*mask)
            
        heatmap = cv2.applyColorMap(mask, color)
        heatmap = np.float32(heatmap) / 255.0
        cam = heatmap / np.max(heatmap)
        
        return np.uint8(255 * cam)

            
    def mask_compute(self, attention_list):
        
        result_mask_list = []
        
        for attention in attention_list:
            batch, channels, height, width = attention.size()
            print(channels)
            print(height)
            print(width)
            
            result = torch.eye(n=attention.size()[1])




if __name__ == '__main__':
    attn_visualizer = AttentionMapVisualizing(head_fusion='mean',
                                              discard_ratio=0.8)
    
    # input = torch.ones((1, 3, 256, 512))
    image = Image.open('dataset_root/Dataset/Enhancement/Underwater/SeaThru/D5/raw_tmp/LFT_3375.png')
    image = np.array(image) / 255.0
    image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    
    input = Image.open('dataset_root/Dataset/Enhancement/Underwater/SeaThru/D5/depth_tmp/depthLFT_3375.png')
    input = np.array(input) / 1000.0
    input = torch.from_numpy(input).unsqueeze(0)
    
    print('image.shape: ',image.size())  
    print('input.shape: ',input.size())
    atten_mask_list = attn_visualizer([input])
    print('atten_mask_list[0]: ',atten_mask_list[0].shape)  

    
    image = attn_visualizer.show_mask_on_image(img=image, mask=atten_mask_list[0])
    
    cv2.imshow("hihih", image)
    cv2.waitKey(0)
    