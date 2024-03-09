
import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.network_builder import MODEL_BUILDER
from dataset.dataload_builder import DATALOAD_BUILDER
import tqdm

from matplotlib import pyplot as plt
from collections import OrderedDict

from PIL import Image

from utils.image_processing import normalize_result, uw_inv_normalize
from utils.research_visualizing import AttentionMapVisualizing

class Joint_Model_Test(object):
    def __init__(self, opt):
        self.opt = opt
    
    def device_initialize(self, 
                          device='', 
                          batch_size=1):
        
        device = str(device).strip().lower().replace('cuda:', '').strip()  # to string, 'cuda:0' to '0'
        cpu = device == 'cpu'
        
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            torch.cuda.empty_cache()        # 언제나 GPU를 한번 씩 비워주자.
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
            assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
                f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

        cuda_flag = not cpu and torch.cuda.is_available()

        if cuda_flag:
            devices = device.split(',') if device else '0'
            space1 = ' ' * 5
            print(space1+ f"devices: {devices}")

            n = len(devices)
            if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
                assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
            space2 = ' ' * 10
            for i, d in enumerate(devices):
                p = torch.cuda.get_device_properties(i)
                print(f"{space2}🚀 CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)")  # bytes to MB
            
            return devices
        else:
            print('🚀 CPU is used!')
            return device        
        
    def get_num_lines(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines) 


    def test(self):
        """Test function."""
        space1 = " "*5 
        space2 = " "*10
        
        print("\n🚀🚀🚀 Setting Gpu before Test! 🚀🚀🚀")
        device =  self.device_initialize(device=self.opt.device, batch_size=self.opt.batch_size)

        print("\n🚀🚀🚀 Setting Model for Test!! 🚀🚀🚀")
        model = MODEL_BUILDER.build(self.opt.model_cfg)
        
        if device != 'cpu':
            device = int(device[0])

        self.opt.test_dataloader_cfg['multiprocessing_distributed'] = False
        dataloader = DATALOAD_BUILDER.build(self.opt.test_dataloader_cfg)
                    
        if device != 'cpu':
            loc = 'cuda:{}'.format(device)
            checkpoint = torch.load(self.opt.test_checkpoint_path, map_location = loc)
            model.load_state_dict(checkpoint['model'])
            model.to('cuda:{}'.format(device))
        else:
            checkpoint = torch.load(self.opt.test_checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])      
            
        model.eval()    

        with open(self.opt.test_txt_file, 'r') as f:
            image_lines = f.readlines()
            num_test_samples = len(image_lines)

        print(space1+"🚀 now testing file name is '{}'".format(self.opt.test_txt_file))
        print(space1+'🚀 now testing {} files with {}'.format(num_test_samples, self.opt.test_checkpoint_path))

        pred_enhanced = []
        pred_depth = []
            
    
        print(space1+'🚀 Try to make directories')
        save_name = 'result_' + self.opt.log_comment
        if not os.path.exists(os.path.dirname(save_name)):
            try:
                sucess_flag = True
                os.mkdir(save_name)
                os.mkdir(save_name + '/enhanced_output')
                os.mkdir(save_name + '/depth_output')
                os.mkdir(save_name + '/depth_cmap')
                    
                if self.opt.is_save_input_image is True:
                    os.mkdir(save_name + '/input')
                
                if self.opt.is_save_gt_image is True:
                    os.mkdir(save_name + '/enhanced_gt')
                    os.mkdir(save_name + '/depth_gt')
                    os.mkdir(save_name + '/depth_gt_cmap')
                    
            except OSError as e:
                sucess_flag = False
                raise ValueError("{} is must be not exist.".format(os.path.dirname(save_name)))
            if sucess_flag:
                print(space2+"making the directories is successful!")
        
        start_time = time.time()
        
        print("\n🚀🚀🚀 Start Test!! 🚀🚀🚀")
        print(space1+"🚀 Precdicting Inputs...")
        with torch.no_grad():
            for idx, sample in tqdm.tqdm(enumerate(dataloader.data)):
                if device != 'cpu':
                    image = torch.autograd.Variable(sample['image'].cuda())
                    if self.opt.is_save_gt_image is True:    
                        gt_enhanced = torch.autograd.Variable(sample['enhanced'].cuda())
                        gt_depth = torch.autograd.Variable(sample['depth'].cuda())
                else:
                    image = torch.autograd.Variable(sample['image']).to(torch.device("cpu"))
                    if self.opt.is_save_gt_image is True:
                        gt_enhanced = torch.autograd.Variable(sample['enhanced']).to(torch.device("cpu"))
                        gt_depth = torch.autograd.Variable(sample['depth']).to(torch.device("cpu"))
                
                depth_est, enhanced_est = model(image)

                
                
                if self.opt.is_save_input_image is True:
                    image_raw = uw_inv_normalize(image[0]).cpu().numpy().transpose(1,2,0)
                    
                if self.opt.is_save_gt_image is True:
                    gt_enhanced_raw = gt_enhanced[0].cpu().numpy().transpose(1,2,0)
                    gt_depth_raw = gt_depth.cpu().numpy().squeeze()
                    
                pred_enhanced = enhanced_est[0].cpu().numpy().transpose(1,2,0)
                pred_depth = depth_est.cpu().numpy().squeeze()
                
            
                print("\n🚀🚀🚀 Saving the {}th result..... 🚀🚀🚀".format(idx))

                save_image_tag = str(idx) + '_' + image_lines[idx].split()[0]
                # save_image_tag = image_lines[idx].split()[0]

                if self.opt.is_save_gt_image is True:
                    # gt_enhanced_tag = image_lines[idx].split()[1]
                    # gt_depth_tag = image_lines[idx].split()[2]
                    gt_enhanced_tag = str(idx) + '_' + image_lines[idx].split()[1]
                    gt_depth_tag = str(idx) + '_' + image_lines[idx].split()[2]
                    depth_scaling = image_lines[idx].split()[3]   

                filename_pred_png = save_name + '/enhanced_output/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png') 
                
                pred_enhanced_scaled = pred_enhanced * 255.0
                pred_enhanced_scaled = pred_enhanced_scaled.astype(np.uint8) 
                Image.fromarray(pred_enhanced_scaled).save(filename_pred_png)

                    
                filename_pred_png = save_name + '/depth_output/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png')
                
                pred_depth_scaled = pred_depth * float(1000)
                pred_depth_scaled = pred_depth_scaled.astype(np.uint16)         
                Image.fromarray(pred_depth_scaled).save(filename_pred_png)

                ##################################
                vmin = pred_depth[pred_depth>0.001].min()
                vmax = pred_depth.max()
                tmp = np.zeros_like(pred_depth) + vmin
                tmp[pred_depth>vmin] = pred_depth[pred_depth>vmin]
                tmp[pred_depth<=vmin] = vmax
                ##################################
                
                attn_visualizer = AttentionMapVisualizing(head_fusion='mean', discard_ratio=0.8)
                
                filename_cmap_png = save_name + '/depth_cmap/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png')
                atten_mask_list = attn_visualizer([depth_est])
                
                image = attn_visualizer.show_mask(mask=1/np.array(atten_mask_list[0].cpu()), color=cv2.COLORMAP_INFERNO)
                cv2.imwrite(filename_cmap_png, image)       
                
                
                if self.opt.is_save_gt_image is True:
                    filename_gt_png = save_name + '/enhanced_gt/' + '_' + gt_enhanced_tag.replace('/','_').replace('.jpg', '.png')
                    
                    gt_enhanced_scaled = gt_enhanced_raw * 255.0
                    gt_enhanced_scaled = gt_enhanced_scaled.astype(np.uint8)
                    pred_enhanced_scaled = Image.fromarray(pred_enhanced_scaled)
                    gt_enhanced_scaled = Image.fromarray(gt_enhanced_scaled)

                    merged_enhanced = Image.new("RGB", (pred_enhanced_scaled.width, pred_enhanced_scaled.height*2))
                    merged_enhanced.paste(gt_enhanced_scaled, (0, 0))
                    merged_enhanced.paste(pred_enhanced_scaled, (0, pred_enhanced_scaled.height))
                    merged_enhanced.save(filename_gt_png)
                    
                                    
                    filename_gt_png = save_name + '/depth_gt/' + '_' + gt_depth_tag.replace('/','_').replace('.jpg', '.png')
                    
                    gt_depth_scaled = gt_depth_raw * float(depth_scaling)
                    gt_depth_scaled = gt_depth_scaled.astype(np.uint16)
                    Image.fromarray(gt_depth_scaled).save(filename_gt_png)
                    

                    ###############################

                    vmin = gt_depth_raw[gt_depth_raw>0.001].min()
                    vmax = gt_depth_raw.max()
                    tmp = np.zeros_like(gt_depth_raw) + vmin
                    tmp[gt_depth_raw>vmin] = gt_depth_raw[gt_depth_raw>vmin]
                    tmp[gt_depth_raw<=vmin] = vmax
                    
                    ################################
                    
                    filename_gt_cmap_png = save_name + '/depth_gt_cmap/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png')
                    plt.imsave(filename_gt_cmap_png, np.log10(tmp), cmap='Greys') 
                    # plt.imsave(filename_gt_cmap_png, np.log10(gt_depth_raw[idx]), cmap='Greys')   
                    
                            
                if self.opt.is_save_input_image is True:                      
                    filename_input_png = save_name + '/input/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png')
                    input_image = image_raw * 255.0
                    input_image = input_image.astype(np.uint8)
                    Image.fromarray(input_image).save(filename_input_png)
                    
            print("\n🚀🚀🚀 Testing is Ended..... 🚀🚀🚀")