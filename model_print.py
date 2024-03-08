from email.policy import strict
import torch
import torchvision.transforms as tr
import torch.functional as F
import torchsummaryX
import torch.nn as nn
import os
import warnings
import time
import tqdm


from core.evaluate.evaluation_builder import EVALUATOR_BUILDER
from core.loss.loss_builder import LOSS_BUILDER
from core.models.network_builder import MODEL_BUILDER
from core.optimizer.optimizer import OPTIMIZER
from dataset.dataload_builder import DATALOAD_BUILDER

from PIL import Image
import numpy as np

depth_range = dict(min_depth_eval=0.001,
                   max_depth_eval=80,
                   max_depth=80
                   )


train_parm = dict(num_threads=4,
                  batch_size=6,
                  num_epochs=50,
                  checkpoint_path=None,
                  retrain=False,
                  do_cosine_scheduler=False,
                  do_mixed_precison=False,
                  )

freq = dict(save_freq=30,
            log_freq=50,
            eval_freq=500
            )


etc = dict(do_online_eval=True,
           do_use_logger=True,
           is_checkpoint_save=True
           )


# Log & Save Setting
log_save_cfg = dict(log_directory='save/summaries/log',
                    eval_log_directory='save/summaries/eval',
                    model_save_directory ='save/checkpoints',
                    wandb_save_path = 'dataset_root/wandb'
                    )


# Dataset
dataset = dict(train_data_path='dataset_root/Dataset/KITTI/input',
               train_gt_path='dataset_root/Dataset/KITTI/label_depth',
               eval_data_path='dataset_root/Dataset/KITTI/input',
               eval_gt_path='dataset_root/Dataset/KITTI/label_depth',
               test_data_path='dataset_root/Dataset/KITTI/input',
               test_gt_path='dataset_root/Dataset/KITTI/label_depth',
               
               train_txt_file='dataset/cfg/kitti/kitti_train_dataset.txt',
               eval_txt_file='dataset/cfg/kitti/kitti_test_dataset.txt',
               test_txt_file='dataset/cfg/kitti/kitti_test_dataset.txt'
               )



image_size = dict(input_height=256,
                  input_width=256
                  )



# model_cfg = dict(type='Build_Structure',
#                                   img_size=(image_size['input_height'], image_size['input_width']),
#                                   structure_cfg = dict(type='UIEC2Net')
#                                   ),


# model_cfg = dict(type='Build_OtherModels',
#                  structure_cfg=dict(type='UDepth',
#                                     n_bins=80, 
#                                     min_val=0.001,
#                                     max_val=20, 
#                                     norm='linear')
#                  )


# model_cfg = dict(type='Build_OtherModels',
#                  structure_cfg = dict(type='BTS',
#                                     max_depth = depth_range['max_depth'],
#                                     bts_size= 512)
#                  )



# model_cfg = dict(type='Build_OtherModels',
#                  structure_cfg = dict(type='GLPDepth',
#                                       max_depth = depth_range['max_depth'],
#                                       is_train= False)
#                  )


# model_cfg = dict(type='Build_OtherModels',  
#                 structure_cfg = dict(type='CC_Module')
#                 )


# model_cfg = dict(type='Build_OtherModels',  
#                 structure_cfg = dict(type='GeneratorFunieGAN')
#                 )



# model_cfg = dict(type='Build_OtherModels',  
#                 structure_cfg = dict(type='UGAN_Generator')
#                 )


# model_cfg = dict(type='Build_OtherModels',  
#                 structure_cfg = dict(type='UIE_DAL')
#                 )



# model_cfg = dict(type='Build_OtherModels',  
#                 structure_cfg = dict(type='UIEC2Net')
#                 )


# model_cfg = dict(type='Build_OtherModels',  
#                 structure_cfg = dict(type='Generator')
#                 )


# model_cfg = dict(type='Build_OtherModels',  
#                  structure_cfg = dict(type='UWCNN',
#                                       get_parameter=True)
# )


# model_cfg = dict(type='Build_OtherModels',  
#                 structure_cfg = dict(type='NewCRFDepth',
#                                     version = 'large07',
#                                     inv_depth=False,
#                                     max_depth=depth_range['max_depth']
#                                     )
#                 )


# model_cfg = dict(type='Build_EncoderDecoder',
#                  img_size=(image_size['input_height'], image_size['input_width']),
#                  backbone_pretrained_path=None,
#                  strict=True,
#                  encoder_cfg = dict(type='eh_mit2_b3'),
#                  decoder_cfg = dict(type='joint_de_single13_4_head6_b3_2'),
#                  task_cfg = dict(type='Enhancement_Task')
#                  )



# model_cfg = dict(type='Build_EncoderDecoder',
#                  img_size=(image_size['input_height'], image_size['input_width']),
#                  backbone_pretrained_path=None,
#                  strict=True,
#                  encoder_cfg = dict(type='eh_mit2_b3'),
#                  decoder_cfg = dict(type='joint_eh_single13_head6_b3_2'),
#                  task_cfg = dict(type='Enhancement_Task')
#                  )

model_cfg = dict(type='Build_Structure',
                img_size=(image_size['input_height'], image_size['input_width']),
                structure_cfg = dict(type='Joint_ID',
                                    de_checkpoint=None,
                                    de_strict=True,
                                    eh_checkpoint=None,
                                    eh_strict=True,
                                    is_de_no_grad=False,
                                    is_eh_no_grad=False,
                                    depth_model_cfg = dict(backbone_pretrained_path=None,
                                                            strict=True,
                                                            encoder_cfg = dict(type='de_mit_pp_b3'),
                                                            decoder_cfg = dict(type='joint_de_single_head'),
                                                            task_cfg = dict(type='DepthEstimation_Task',
                                                                            max_depth=depth_range['max_depth'])
                                                            ),
                                    enhanced_model_cfg = dict(backbone_pretrained_path=None,
                                                                strict=True,
                                                                encoder_cfg = dict(type='eh_mit_pp_b3'),
                                                                decoder_cfg = dict(type='joint_eh_single_head'),
                                                                task_cfg = dict(type='Enhancement_Task')
                                                                )
                                    )
                )




if __name__ == '__main__':

    model = MODEL_BUILDER.build(model_cfg).to('cuda:0')
    # model = MODEL_BUILDER.build(model_cfg)
    iter_num = 500
    
    model.eval()
      
    start = time.time()   
    with torch.no_grad():
        for i in tqdm.tqdm(range(iter_num)):
            # output1 = model(torch.rand(1, 3, 256, 256))
            output1 = model(torch.rand(1, 3, 256, 256).to(torch.device('cuda:0')))
            # output1, output2 = model(torch.rand(1, 3, 480, 640).to(torch.device('cuda:0')))
        elapsed_time = time.time()-start
    # print('output1.shape: ', output1.shape)
    # print('output2.shape: ', output2.shape)
    print("Elapesed time: '{} sec' for '{} files' -> '{} Hz'".format(str(elapsed_time), iter_num, iter_num/elapsed_time))
    
    # torchsummaryX.summary(model, torch.rand(1, 3, 352, 1216).to('cuda:0'))
    
    num_params_update = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {} M".format(num_params_update/1000000.0))
    num_params_update = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {} M".format(num_params_update/1000000.0))
    
    # Total number of parameters: 72.040307 / 32 hz / 8.5 hz
    # Total number of parameters: 150.2489  / 15.1 hz / 2.9 hz 
