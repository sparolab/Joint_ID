
import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tr

from core.models.network_builder import MODEL_BUILDER
from dataset.dataload_builder import DATALOAD_BUILDER
import tqdm

from matplotlib import pyplot as plt
from PIL import Image

from utils.image_processing import normalize_result, uw_inv_normalize
from utils.research_visualizing import AttentionMapVisualizing

class Joint_Model_Video_Test(object):
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


    def video_test(self):
        """Test function."""
        space1 = " "*5 
        space2 = " "*10
        
        print("\n🚀🚀🚀 Setting Gpu before Test! 🚀🚀🚀")
        device =  self.device_initialize(device=self.opt.device, batch_size=self.opt.batch_size)

        print("\n🚀🚀🚀 Setting Model for Test!! 🚀🚀🚀")
        model = MODEL_BUILDER.build(self.opt.model_cfg)
                 
        if device != 'cpu':
            device = int(device[0])
            loc = 'cuda:{}'.format(device)
            checkpoint = torch.load(self.opt.test_checkpoint_path, map_location = loc)
            model.load_state_dict(checkpoint['model'])
            model.to('cuda:{}'.format(device))
        else:
            checkpoint = torch.load(self.opt.test_checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])      
            
        model.eval()    

        normalizer = tr.Normalize(mean=[0.13553666, 0.41034216, 0.34636855], std=[0.04927989, 0.10722694, 0.10722694])
        
        cap = cv2.VideoCapture(self.opt.video_test_cfg['video_txt_file'])
        origin_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        origin_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        
        if self.opt.video_test_cfg['auto_crop'] is True:
            revised_width = 32 * (origin_width // 32)
            revised_height = 32 * (origin_height // 32)
            
            top_margin = int((origin_height - revised_height) / 2) 
            left_margin = int((origin_width - revised_width) / 2)     

            bottom_margin = int(top_margin + revised_height)
            right_margin = int(left_margin + revised_width)
            
            do_resize_crop = False
        else:
            revised_width = self.opt.video_test_cfg['img_size'][1]
            revised_height = self.opt.video_test_cfg['img_size'][0]
            revised_width = 32 * (revised_width // 32)
            revised_height = 32 * (revised_height // 32)
            
            if self.opt.video_test_cfg['do_resize_crop'] is True:
                do_resize_crop = True
            
            else:
                if self.opt.video_test_cfg['do_center_crop'] is True:
                    top_margin = int((origin_height - revised_height) / 2) 
                    left_margin = int((origin_width - revised_width) / 2) 
                    
                    bottom_margin = int(top_margin + revised_height)
                    right_margin = int(left_margin + revised_width)
                    
                    do_resize_crop = False
                      

        print(space1+"🚀 now testing file name is '{}'".format(self.opt.video_test_cfg['video_txt_file']))
        print(space1+'🚀 Try to make directories')
        save_name = 'video_result_' + self.opt.log_comment
        # if not os.path.exists(os.path.dirname(save_name)):
        #     try:
        #         sucess_flag = True
        #         os.mkdir(save_name)
        #         os.mkdir(save_name + '/enhanced_output')
        #         os.mkdir(save_name + '/depth_cmap')
                                                    
        #     except OSError as e:
        #         sucess_flag = False
        #         raise ValueError("{} is must be not exist.".format(os.path.dirname(save_name)))
        #     if sucess_flag:
        #         print(space2+"making the directories is successful!")

        prev_time = time.time() 
        total_frames = 0
        pred_enhanced = []
        pred_depth = []
        fps_list = []
        print("\n🚀🚀🚀 Start Test!! 🚀🚀🚀")
        print(space1+"🚀 Precdicting Inputs...")
        with torch.no_grad():
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break            
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                                
                if do_resize_crop:
                    frame = cv2.resize(frame, (revised_width, revised_height), interpolation=cv2.INTER_CUBIC)
                else:
                    frame = frame[top_margin:bottom_margin, left_margin:right_margin, :]
                    
                img = torch.from_numpy(frame.transpose((2, 0 ,1))).float()
                img = normalizer(img).unsqueeze(0)
                
                depth_est, enhanced_est, attn_depth = model(img.cuda())
                pred_enhanced = enhanced_est[0].cpu().numpy().transpose(1,2,0)


                current_time = time.time()       
                elapsed_time = current_time - prev_time
                prev_time = current_time
                fps = 1/(elapsed_time)
                str = "FPS: %0.2f" % fps
                pred_enhanced = cv2.cvtColor(pred_enhanced, cv2.COLOR_RGB2BGR)
                
                cv2.putText(pred_enhanced, str, (revised_width//20, revised_height//10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
                cv2.imshow("enhanced_frame", pred_enhanced)
                cv2.waitKey(1)
        
            cap.release()
            cv2.destroyAllWindows()

        # with torch.no_grad():
        #     for idx, sample in tqdm.tqdm(enumerate(dataloader.data)):
        #         if device != 'cpu':
        #             image = torch.autograd.Variable(sample['image'].cuda())
        #             if self.opt.is_save_gt_image is True:    
        #                 gt_enhanced = torch.autograd.Variable(sample['enhanced'].cuda())
        #                 gt_depth = torch.autograd.Variable(sample['depth'].cuda())
        #         else:
        #             image = torch.autograd.Variable(sample['image']).to(torch.device("cpu"))
        #             if self.opt.is_save_gt_image is True:
        #                 gt_enhanced = torch.autograd.Variable(sample['enhanced']).to(torch.device("cpu"))
        #                 gt_depth = torch.autograd.Variable(sample['depth']).to(torch.device("cpu"))
                
        #         depth_est, enhanced_est, attn_depth = model(image)
                
        #         # filter_ = GF(enhanced_est, enhanced_est)
                
        #         # enhanced_est = enhanced_est + 1.25*(enhanced_est - filter_)
        #         # enhanced_est[enhanced_est>1.0] = 1.0
        #         # enhanced_est[enhanced_est<0.0001] = 0.0001
                
                
        #         if self.opt.is_save_attn is True:
        #             attn_map_list = attn_depth
                
        #         if self.opt.is_save_input_image is True:
        #             image_raw = uw_inv_normalize(image[0]).cpu().numpy().transpose(1,2,0)
                    
        #         if self.opt.is_save_gt_image is True:
        #             gt_enhanced_raw = gt_enhanced[0].cpu().numpy().transpose(1,2,0)
        #             gt_depth_raw = gt_depth.cpu().numpy().squeeze()
                    
        #         pred_enhanced = enhanced_est[0].cpu().numpy().transpose(1,2,0)
        #         pred_depth = depth_est.cpu().numpy().squeeze()
                
            
        #         print("\n🚀🚀🚀 Saving the {}th result..... 🚀🚀🚀".format(idx))

        #         save_image_tag = str(idx) + '_' + image_lines[idx].split()[0]
        #         # save_image_tag = image_lines[idx].split()[0]

        #         if self.opt.is_save_gt_image is True:
        #             # gt_enhanced_tag = image_lines[idx].split()[1]
        #             # gt_depth_tag = image_lines[idx].split()[2]
        #             gt_enhanced_tag = str(idx) + '_' + image_lines[idx].split()[1]
        #             gt_depth_tag = str(idx) + '_' + image_lines[idx].split()[2]
        #             depth_scaling = image_lines[idx].split()[3]   

        #         filename_pred_png = save_name + '/enhanced_output/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png') 
                
        #         pred_enhanced_scaled = pred_enhanced * 255.0
        #         pred_enhanced_scaled = pred_enhanced_scaled.astype(np.uint8) 
        #         Image.fromarray(pred_enhanced_scaled).save(filename_pred_png)

        #         if self.opt.is_save_attn is True:
        #             channels, height, width = pred_enhanced.shape
                    
        #             for i, attn_map in enumerate(attn_map_list):
        #                 filename_attn_png = save_name + '/attn/' + '_' + save_image_tag.replace('/','_').replace('.jpg','').replace('.png','') 
        #                 filename_attn_png = filename_attn_png + '_' + str(i) + '.png'
                        
        #                 attn_mask_result = attn_visualizer([attn_map])
        #                 # print('attn_mask_result[0].unsqueeze().shape: ', attn_mask_result[0].unsqueeze(dim=0).shape)
        #                 inter_result = F.interpolate(attn_mask_result[0].unsqueeze(dim=0), scale_factor=32/(2**i), mode='nearest')
        #                 # print('inter_result.shape: ', inter_result.shape)
        #                 # print('inter_result[0].cpu().numpy().shape: ', inter_result[0].cpu().numpy().shape)
        #                 inter_result = inter_result.cpu().numpy().transpose(1,2,0)
                                            
        #                 # attn_result = attn_visualizer.show_mask_on_image(pred_enhanced[idx], inter_result)
        #                 attn_result = attn_visualizer.show_mask(inter_result)
        #                 Image.fromarray(attn_result).save(filename_attn_png) 
                
        #         filename_pred_png = save_name + '/depth_output/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png')
        #         ##################################
        #         # vmin = gt_depth_raw[idx][gt_depth_raw[idx]>0.001].min()
                
        #         # pred_depth[idx][gt_depth_raw[idx]<=vmin] = 0.0001
                
        #         # a = np.mean(gt_depth_raw[idx]) / np.mean(pred_depth[idx])
        #         ##################################
                
        #         pred_depth_scaled = pred_depth * float(1000)
        #         pred_depth_scaled = pred_depth_scaled.astype(np.uint16)         
        #         Image.fromarray(pred_depth_scaled).save(filename_pred_png)

                
                
        #         ##################################
        #         vmin = pred_depth[pred_depth>0.001].min()
        #         vmax = pred_depth.max()
        #         tmp = np.zeros_like(pred_depth) + vmin
        #         tmp[pred_depth>vmin] = pred_depth[pred_depth>vmin]
        #         tmp[pred_depth<=vmin] = vmax
        #         ##################################
                
        #         attn_visualizer = AttentionMapVisualizing(head_fusion='mean', discard_ratio=0.8)
                
        #         filename_cmap_png = save_name + '/depth_cmap/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png')
        #         # plt.imsave(filename_cmap_png, np.log10(tmp), cmap='Greys')   

        #         atten_mask_list = attn_visualizer([depth_est])
        #         # atten_mask_list = attn_visualizer([np.expand_dims(pred_depth, axis=0)])
                
        #         image = attn_visualizer.show_mask(mask=1/np.array(atten_mask_list[0].cpu()), color=cv2.COLORMAP_INFERNO)
        #         cv2.imwrite(filename_cmap_png, image)       
                
                
        #         if self.opt.is_save_gt_image is True:
        #             filename_gt_png = save_name + '/enhanced_gt/' + '_' + gt_enhanced_tag.replace('/','_').replace('.jpg', '.png')
                    
        #             gt_enhanced_scaled = gt_enhanced_raw * 255.0
        #             gt_enhanced_scaled = gt_enhanced_scaled.astype(np.uint8)
        #             pred_enhanced_scaled = Image.fromarray(pred_enhanced_scaled)
        #             gt_enhanced_scaled = Image.fromarray(gt_enhanced_scaled)

        #             merged_enhanced = Image.new("RGB", (pred_enhanced_scaled.width, pred_enhanced_scaled.height*2))
        #             merged_enhanced.paste(gt_enhanced_scaled, (0, 0))
        #             merged_enhanced.paste(pred_enhanced_scaled, (0, pred_enhanced_scaled.height))
        #             merged_enhanced.save(filename_gt_png)
                    
                                    
        #             filename_gt_png = save_name + '/depth_gt/' + '_' + gt_depth_tag.replace('/','_').replace('.jpg', '.png')
                    
        #             gt_depth_scaled = gt_depth_raw * float(depth_scaling)
        #             gt_depth_scaled = gt_depth_scaled.astype(np.uint16)
        #             Image.fromarray(gt_depth_scaled).save(filename_gt_png)
                    

        #             ###############################

        #             vmin = gt_depth_raw[gt_depth_raw>0.001].min()
        #             vmax = gt_depth_raw.max()
        #             tmp = np.zeros_like(gt_depth_raw) + vmin
        #             tmp[gt_depth_raw>vmin] = gt_depth_raw[gt_depth_raw>vmin]
        #             tmp[gt_depth_raw<=vmin] = vmax
                    
        #             # tmp[np.isnan(gt_raw[idx])] = vmin
        #             # tmp[np.isinf(gt_raw[idx])] = vmax
        #             # tmp = tmp*(pred_depths[idx].max()/vmax)

        #             ################################
                    
        #             filename_gt_cmap_png = save_name + '/depth_gt_cmap/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png')
        #             plt.imsave(filename_gt_cmap_png, np.log10(tmp), cmap='Greys') 
        #             # plt.imsave(filename_gt_cmap_png, np.log10(gt_depth_raw[idx]), cmap='Greys')   
                    
                            
        #         if self.opt.is_save_input_image is True:                      
        #             filename_input_png = save_name + '/input/' + '_' + save_image_tag.replace('/','_').replace('.jpg', '.png')
        #             input_image = image_raw[idx] * 255.0
        #             input_image = input_image.astype(np.uint8)
        #             Image.fromarray(input_image).save(filename_input_png)
                    
        #     print("\n🚀🚀🚀 Testing is Ended..... 🚀🚀🚀")
            