

import numpy as np

import os
import time
import tqdm

import argparse
import torch

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import yaml

from tensorboardX import SummaryWriter
from utils.logger import Wandb_Logger, TensorBoardLogger
import sys
from utils.image_processing import normalize_result, inv_normalize, uw_inv_normalize
from mmcv.utils import Config

from core.evaluate.evaluation_builder import EVALUATOR_BUILDER
from core.loss.loss_builder import LOSS_BUILDER
from core.models.network_builder import MODEL_BUILDER
from core.optimizer.optimizer import OPTIMIZER
from dataset.dataload_builder import DATALOAD_BUILDER
from core.optimizer.scheduler import SCHEDULER


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class Joint_Model_Train(object):
    def __init__(self, opt: argparse.Namespace):
        self.opt = opt

    def checkpoint_loader(self, 
                          checkpoint_path, 
                          model,
                          device,
                          do_retrain = False
                          ):
        space1 = "".rjust(5)
        space2 = "".rjust(10)        
        
        model_loaded_sucess = False
        global_step = 0
        
        if os.path.isfile(checkpoint_path):
            print(space1+"🚀 Start Loading checkpoint '{}'".format(checkpoint_path))
            
            if device is None:
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else: 
                loc = 'cuda:{}'.format(device)
                checkpoint = torch.load(checkpoint_path, map_location= loc)
                model.module.load_state_dict(checkpoint['model'])
            
            global_step = checkpoint['global_step']
            
            print(space1+"🚀 Loaded checkpoint '{}' (global_step {})".format(checkpoint_path, checkpoint['global_step']))
            model_loaded_sucess = True
        else:
            print(space1+"🚀 No checkpoint found at '{}'".format(checkpoint_path))
        if do_retrain:
            print(space1+"🚀 This Checkpoint model is retrained!")
            global_step = 0
        return {'model':model, 'global_step':global_step, 'load_sucess':model_loaded_sucess}
    
    
    def main_worker(self,
                    process,
                    ngpus_per_node, 
                    opt
                    ):
        space1, space2 = " "*5, " "*10
        
        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and process == 0):
            print("\n🚀🚀🚀 Setting Model for Training!!  🚀🚀🚀")
        
        opt.gpu = None if process == 'cpu' else process

        if opt.gpu is not None:  
            torch.cuda.set_device(opt.gpu)
            print(space1+"Use GPU: {} for training".format(opt.gpu)) 
        else:
            print(space1+"Use CPU for training")
        
        ### Model Setting ###
        model = MODEL_BUILDER.build(opt.model_cfg)
        
        if  opt.gpu is not None and opt.multiprocessing_distributed:
            if opt.dist_url == "env://" and opt.rank == -1:
                opt.rank = int(os.environ["RANK"])
               
            opt.rank = opt.rank * ngpus_per_node + process
            print(space1+"dist_backend : {}, dist_url: {}, world_size: {}, rank: {}".format(opt.dist_backend,
                                                                                            opt.dist_url,
                                                                                            opt.world_size,
                                                                                            opt.rank))
            dist.init_process_group(backend= opt.dist_backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
            opt.batch_size = int(opt.batch_size / ngpus_per_node)
            print(space1+"🚀 gpu: {},     batch_size per GPU: {},     ngpus_per_node: {}".format(opt.gpu, opt.batch_size, ngpus_per_node))
            torch.cuda.set_device(opt.gpu)
            model = model.to(opt.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters= True)    
        elif opt.gpu is not None:
            model = torch.nn.DataParallel(model, device_ids=[opt.gpu])    
        
        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
            num_params = sum([np.prod(p.size()) for p in model.parameters()])
            print(space1+"Total number of parameters: {}".format(num_params))
            num_params_update = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
            print(space1+"Total number of learning parameters: {}".format(num_params_update))

        ### Optimizer Setting ###
        if opt.gpu is not None:
            opt.optimizer_cfg['params'] = model.module.parameters()
            optimizer = OPTIMIZER.build(opt.optimizer_cfg)
        
        else:
            opt.optimizer_cfg['params'] = model.parameters()
            optimizer = OPTIMIZER.build(opt.optimizer_cfg)        

        ### Criterion Setting ###

        criterion = LOSS_BUILDER.build(opt.loss_build_cfg)
        
        self.main_train(model, optimizer, criterion, opt, opt.gpu, ngpus_per_node)
        
    def main_train(self,
                   model,
                   optimizer,
                   criterion,
                   opt, 
                   device, 
                   ngpus_per_node):
        space1, space2 = " "*5, " "*10

        cudnn.benchmark = True
        
        ### Etc Setting ###
        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
            print("\n🚀🚀🚀 Setting etc... for Training!!  🚀🚀🚀")
            if opt.do_use_logger == 'Wandb':
                param = Config.fromfile(opt.config)
                train_logger = Wandb_Logger(opt, ngpus_per_node, param, f"{opt.log_comment}_train")     
            
            elif opt.do_use_logger == 'Tensorboard':
                # log_dir = os.path.join(opt.log_directory, opt.log_comment) 
                log_dir = opt.log_directory
                train_logger = TensorBoardLogger(log_dir=log_dir, comment=f"{opt.log_comment}_train", flush_sec=30)
    
        if opt.checkpoint_path is not None:
            checkpoint = self.checkpoint_loader(opt.checkpoint_path, model, device)
            model = checkpoint['model']
            global_step = checkpoint['global_step']
            model_loaded_sucess = checkpoint['load_sucess']
            
        else:
            if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
                print(space1+"🚀 We don't use checkpoint! ")
            model_loaded_sucess = False
            global_step = 0
        
        if opt.gpu is None:
            model = model.to(torch.device("cpu"))
            
        if opt.scheduler_cfg is not None:
            opt.scheduler_cfg['optimizer'] = optimizer
            train_scheduler = SCHEDULER.build(opt.scheduler_cfg)
        else:
            end_learning_rate = 0.01 * opt.optimizer_cfg['lr']

        ### Dataloader Setting ###
        if opt.multiprocessing_distributed:
            opt.train_dataloader_cfg['multiprocessing_distributed'] = True
            opt.eval_dataloader_cfg['multiprocessing_distributed'] = True
        else:
            opt.train_dataloader_cfg['multiprocessing_distributed'] = False
            opt.eval_dataloader_cfg['multiprocessing_distributed'] = False 
            
        dataloader = DATALOAD_BUILDER.build(opt.train_dataloader_cfg)
        
        # checkpoint_path = os.path.join(opt.model_save_directory, opt.log_comment)
        checkpoint_path = opt.model_save_directory
        
        ### Evaluator Setting ###
        if opt.do_online_eval:
            dataloader_eval = DATALOAD_BUILDER.build(opt.eval_dataloader_cfg)
            
            opt.evaluator_cfg['device'] = device
            opt.evaluator_cfg['ngpus'] = ngpus_per_node
            opt.evaluator_cfg['dataloader_eval'] = dataloader_eval
            opt.evaluator_cfg['save_dir'] = opt.model_save_directory

            evaluator = EVALUATOR_BUILDER.build(opt.evaluator_cfg)
                 
        start_time = time.time()
        duration = 0

        steps_per_epoch = len(dataloader.data)
        num_total_steps = opt.num_epochs * steps_per_epoch
        epoch = global_step // steps_per_epoch
        
        print(space1+"🚀 epochs: {} / steps_per_epoch: {} / num_total_steps: {}".format(opt.num_epochs, steps_per_epoch, num_total_steps))
        
        if opt.do_use_logger == 'Wandb' and (not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0)):
            train_logger.logging_model_watch(model, criterion, opt.log_freq)
        
        elif opt.do_use_logger == 'Tensorboard':
            train_logger.model_graph_summary(model, torch.rand(1, 3, opt.input_height, opt.input_width))

        if opt.do_mixed_precison == True:
            scaler = torch.cuda.amp.GradScaler() 

        loss_tag_list = criterion.loss_tag_list
        
        print("\n🚀🚀🚀 Start Training!!  🚀🚀🚀")
        while epoch < opt.num_epochs:
            if opt.multiprocessing_distributed == True:
                dataloader.train_sampler.set_epoch(epoch)
                
            for step, sample_batched in tqdm.tqdm(enumerate(dataloader.data), desc=f"Epoch: {epoch}/{opt.num_epochs}. Loop: Train", 
                                                  total=len(dataloader.data)) if opt.rank == 0 or not opt.multiprocessing_distributed else enumerate(dataloader.data):
                optimizer.zero_grad()
                before_op_time = time.time()                
                
                if opt.distributed == True:
                    image = torch.autograd.Variable(sample_batched['image'].cuda(device, non_blocking=True))
                    depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(device, non_blocking=True))
                    enhanced_gt = torch.autograd.Variable(sample_batched['enhanced'].cuda(device, non_blocking=True))
                else:
                    image = torch.autograd.Variable(sample_batched['image'].to(torch.device("cpu")))
                    depth_gt = torch.autograd.Variable(sample_batched['depth'].to(torch.device("cpu")))
                    enhanced_gt = torch.autograd.Variable(sample_batched['enhanced'].to(torch.device("cpu")))
                
                if opt.do_mixed_precison == True:
                    with torch.cuda.amp.autocast():
                        depth_est, enhanced_est = model(image)                     
                        loss = criterion.forward([enhanced_est, depth_est], [enhanced_gt, depth_gt])
                        
                        final_loss = loss['final']
                        loss_value_list = loss['value_list']
                        
                    scaler.scale(final_loss).backward()                 
                    scaler.step(optimizer)
                    scaler.update()
                                                                              
                else:
                    # depth_est = model(image)
                    depth_est, enhanced_est = model(image)
                    
                    loss = criterion.forward([enhanced_est, depth_est], [enhanced_gt, depth_gt])   
                                                             
                    final_loss = loss['final']
                    loss_value_list = loss['value_list']                
                    
                    final_loss.backward()
                    optimizer.step() 
                
                if opt.scheduler_cfg is None:
                    for param_group in optimizer.param_groups:
                        current_lr = (opt.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                        param_group['lr'] = current_lr
                else:
                    train_scheduler.step(global_step / steps_per_epoch)
                
                current_lr = optimizer.param_groups[0]['lr']

                # logging for terminal
                if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, final_loss))
                    if np.isnan(final_loss.cpu().item()):
                        print('NaN in loss occurred. Aborting training.')
                
                # logging2 for terminal
                duration += time.time() - before_op_time
                if global_step and global_step % opt.log_freq == 0 and not model_loaded_sucess:
                    examples_per_sec = opt.batch_size / duration * opt.log_freq
                    duration = 0
                    time_sofar = (time.time() - start_time) / 3600
                    training_time_left = (num_total_steps / global_step - 1.0) * time_sofar

                    print_string = space1+"🚀 GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h"
                    print(print_string.format(opt.gpu, examples_per_sec, final_loss, time_sofar, training_time_left))
                    
                    # logging for Wandb
                    if opt.do_use_logger == 'Wandb' and (not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0)):
                        train_logger.logging_graph(loss_tag_list, loss_value_list, global_step)                 
                        train_logger.logging_graph(['learning_rate'], [current_lr], global_step)

                        depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                        depth_image_tag_list = ['depth_gt', 'depth_est', 'depth_original']
                        depth_image_list = [normalize_result(1/depth_gt[0, :, :, :].data), normalize_result(1/depth_est[0, :, :, :].data), uw_inv_normalize(image[0, :, :, :]).data]
                        train_logger.logging_images('Output Images', depth_image_list, depth_image_tag_list, global_step)
                        
                        enhanced_image_tag_list = ['enhanced_gt', 'enhanced_est', 'enhanced_original']
                        enhanced_image_list = [enhanced_gt[0, :, :, :].data, enhanced_est[0, :, :, :].data, uw_inv_normalize(image[0, :, :, :]).data]
                        train_logger.logging_images('Output Images', enhanced_image_list, enhanced_image_tag_list, global_step)

                    # logging for Tensorboard
                    if opt.do_use_logger == 'Tensorboard' and (not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0)):
                        train_logger.scalar_summary(loss_tag_list, loss_value_list, global_step)
                        train_logger.scalar_summary(['learning_rate'], [current_lr], global_step)

                        depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                        depth_image_tag_list = ['depth_gt', 'depth_est', 'depth_original']
                        depth_image_list = [normalize_result(1/depth_gt[0, :, :, :].data), normalize_result(1/depth_est[0, :, :, :].data), uw_inv_normalize(image[0, :, :, :]).data]
                        train_logger.image_summary(depth_image_tag_list, depth_image_list, global_step)

                        enhanced_image_tag_list = ['enhanced_gt', 'enhanced_est', 'enhanced_original']
                        enhanced_image_list = [enhanced_gt[0, :, :, :].data, enhanced_est[0, :, :, :].data, uw_inv_normalize(image[0, :, :, :]).data]
                        train_logger.image_summary(enhanced_image_tag_list, enhanced_image_list, global_step)
                
                # Saving models for not do_online_eval
                if opt.is_checkpoint_save and not opt.do_online_eval and global_step and global_step % opt.save_freq == 0:
                    if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0):
                        if opt.distributed:
                            checkpoint = {'global_step': global_step,
                                          'model': model.module.state_dict()
                                          }
                        else:
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict()
                                          }
                            
                        torch.save(checkpoint, opt.model_save_directory + '/joint_model_{}.pth'.format(global_step))
                        print("Sucess to save '{}'.".format(opt.model_save_directory + '/joint_model_{}'.format(global_step)))
                        
                # Saving models for do_online_eval
                if opt.do_online_eval and global_step and global_step % opt.eval_freq == 0 and not model_loaded_sucess:
                    time.sleep(0.1)
                    model.eval()
                    
                    eval_result = evaluator.result_evaluation(opt, model, global_step)
                    error_string = ''
                    
                    if opt.do_use_logger == 'Wandb' and (not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0)):
                        for result in eval_result:
                            error_string = error_string + ' ' + result['error_string']
                            train_logger.logging_graph(result['eval_metrics'], result['eval_measures'], global_step)

                            image_tag_list = result['val_image_tag_list']
                            image_list = result['val_sample']
                            train_logger.logging_images('Val Images', image_list, image_tag_list, global_step)  

                    elif opt.do_use_logger == 'Tensorboard' and (not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0)):
                        for result in eval_result:
                            error_string = error_string + ' ' + result['error_string']
                            train_logger.scalar_summary(result['eval_metrics'], result['eval_measures'], global_step)

                            image_tag_list = result['val_image_tag_list']
                            image_list = result['val_sample']
                            train_logger.image_summary(image_tag_list, image_list, global_step)
                            
                    model.train()   

                model_loaded_sucess = False
                global_step += 1

            epoch += 1
        if opt.do_use_logger == 'Wandb' and (not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0)):
            train_logger.logging_alert("Traing is Ended", error_string) 
            train_logger.logging_train_ended()
        if opt.do_use_logger == 'Tensorboard' and (not opt.multiprocessing_distributed or (opt.multiprocessing_distributed and opt.rank % ngpus_per_node == 0)):
            train_logger.log_flush()
        print("\n🚀🚀🚀 Training is Ended..... 🚀🚀🚀")
        
        
    def device_initialize(self, 
                          device='', 
                          batch_size=0):
        torch.cuda.empty_cache()        # 언제나 GPU를 한번 씩 비워주자.
        
        device = str(device).strip().lower().replace('cuda:', '').strip()  # to string, 'cuda:0' to '0'
        cpu = device == 'cpu'
        
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
            print('torch.cuda.is_available(): ', torch.cuda.is_available())
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

    def base_mkdir_folders(self, dir_path, log_comment):
        num = 0
        mkdir_path = os.path.join(dir_path, log_comment + f'_{num}')
        while os.path.isdir(mkdir_path):
            num = num+1
            mkdir_path = os.path.join(dir_path, log_comment + f'_{num}')
            
        os.mkdir(mkdir_path)
        
        self.opt.model_save_directory = os.path.join(mkdir_path, self.opt.model_save_directory)
        os.mkdir(self.opt.model_save_directory)
        
        self.opt.log_directory = os.path.join(mkdir_path, self.opt.log_directory)
        os.mkdir(self.opt.log_directory)
        
        self.opt.eval_log_directory = os.path.join(mkdir_path, self.opt.eval_log_directory)
        os.mkdir(self.opt.eval_log_directory)

        return num
                         
    def mkdir_logging_checkpoint_folders(self):
        space = "".rjust(5)

        num = self.base_mkdir_folders(self.opt.save_root, self.opt.log_comment)
        self.opt.log_comment = self.opt.log_comment + f'_{num}'

        if self.opt.do_use_logger == 'Wandb':
            mkdir_path = os.path.join(self.opt.wandb_save_path, self.opt.log_comment)
            os.mkdir(mkdir_path)


    def train(self):
        space1 = " "*5 
        space2 = " "*10

        self.mkdir_logging_checkpoint_folders()

        print("\n🚀🚀🚀 Setting Gpu before training! 🚀🚀🚀")
        device =  self.device_initialize(device=self.opt.device, batch_size=self.opt.batch_size)
        ngpus_per_node = torch.cuda.device_count()
        
        self.opt.distributed = True if ngpus_per_node >= 1 else False
            
        if device != 'cpu' and len(device) > 1:
        # if (device != 'cpu' and len(device) > 1):
            self.opt.multiprocessing_distributed = True
            port = np.random.randint(2000, 2345)
            self.opt.dist_url = 'tcp://{}:{}'.format('127.0.0.1', port)

        if self.opt.multiprocessing_distributed:
            self.opt.world_size = ngpus_per_node * self.opt.world_size
            mp.spawn(self.main_worker, nprocs=ngpus_per_node, args= (ngpus_per_node, self.opt))
        else:
            if len(device) == 1:
                device = int(device[0])
            self.main_worker(device, ngpus_per_node, self.opt)            
        
        if self.opt.multiprocessing_distributed:
            dist.destroy_process_group()
                