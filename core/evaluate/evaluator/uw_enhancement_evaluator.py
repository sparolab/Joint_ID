
import torch
import tqdm
import numpy as np
import torch.distributed as dist
import os
import math

from ..evaluation_builder import EVALUATOR
from skimage.metrics import structural_similarity as _ssim
import torchvision.transforms as tr
from ..evaluate_utils.underwater.uiqm_utils import uiqm_compute
from ..evaluate_utils.underwater.uciqe_utils import uciqe_compute

# utils.image_processing import normalize_result, inv_normalize

def inv_normalize(image):
    inv_normal = tr.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normal(image).data

def uw_inv_normalize(image):
    inv_normal = tr.Normalize(
        mean=[-0.13553666/0.04927989, -0.41034216/0.10722694, -0.34636855/0.10722694],
        std=[1/0.04927989, 1/0.10722694, 1/0.10722694]
    )
    return inv_normal(image).data

@EVALUATOR.register_module()
class UW_Enhancement_Evaluator(object):
    def __init__(self,  
                 device,
                 dataloader_eval,
                 ngpus:int=0,
                 save_dir=None,
                 is_checkpoint_save:bool=True
                 ):

        if is_checkpoint_save is True:
            if save_dir is None:
                raise ValueError("If 'is_checkpoint_save' is True, then 'save_dir' is must be not 'False'. but, Got {}".format(save_dir))
        
            if os.path.isdir(save_dir) is False:
                raise ValueError("'save dir' is not exist. but, Got {}".format(save_dir))

        # self.eval_metrics = ['uw_abs_rel', 'uw_log10', 'uw_rms', 'uw_sq_rel', 'uw_log_rms', 'uw_uiqm', 'uw_uciqe' ,'uw_ssim', 'uw_psnr']
        self.eval_metrics = ['uw_abs_rel', 'uw_rms', 'uw_sq_rel', 'uw_ssim', 'uw_psnr']
        self.metrics_len = len(self.eval_metrics)
        
        self.is_checkpoint_save = is_checkpoint_save
        self.checkpoint_dir = save_dir

        self.device = device
        self.ngpus = ngpus
        self.dataloader_eval = dataloader_eval

        self.best_eval_measures_lower_better = torch.zeros(3).cpu() + 1e4
        self.best_eval_measures_higher_better = torch.zeros(2).cpu()
        self.best_eval_steps = np.zeros(5, dtype= np.int32)
        
        self.peeking_num = 0
        
    def enhancement_compute_errors(self, gt, pred):

        rms = (gt - pred) ** 2
        rms = np.sqrt(rms.mean())

        # log_rms = (np.log(gt) - np.log(pred)) ** 2
        # log_rms = np.sqrt(log_rms.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        # err = np.log(pred) - np.log(gt)
        # log10 = np.mean(err)

        pred_t = pred.transpose(1,2,0) * 255.0
        gt_t = gt.transpose(1,2,0) * 255.0

        mse = np.mean((gt_t - pred_t) ** 2)
        psnr = 10 * math.log10(255.0**2/mse)
        
        ssim = _ssim(gt_t, pred_t, multichannel=True)

        # uiqm = uiqm_compute(pred_t)
        # uciqe = uciqe_compute(pred_t)

        # return [abs_rel, log10, rms, sq_rel, log_rms, uiqm, uciqe, ssim, psnr]
        return [abs_rel, rms, sq_rel, ssim, psnr]


    def enhanced_eval(self, opt, model):
        space1 = " "*5
        
        num_metrics = self.metrics_len
        
        if self.device != None:
            eval_measures = torch.zeros(num_metrics + 1).cuda(device=self.device)
        else:
            eval_measures = torch.zeros(num_metrics + 1)
        
        self.val_sample = []
        for idx, eval_sample_batched in tqdm.tqdm(enumerate(self.dataloader_eval.data), 
                                                total=len(self.dataloader_eval.data)) if opt.rank == 0 or not opt.multiprocessing_distributed else enumerate(self.dataloader_eval.data):
            with torch.no_grad():
                image = torch.autograd.Variable(eval_sample_batched['image'].cuda(self.device, non_blocking=True))

                gt_enhanced = eval_sample_batched['enhanced']
                pred_enhanced = model(image)

                pred_enhanced = pred_enhanced.cpu().numpy().squeeze()
                gt_enhanced = gt_enhanced.cpu().numpy().squeeze()
                        
            if idx == self.peeking_num:
                self.val_sample.append(uw_inv_normalize(image[0]))
                self.val_sample.append(pred_enhanced)
                self.val_sample.append(gt_enhanced)
            
            gt_enhanced[gt_enhanced < 0.00001] = 0.00001

            measures = self.enhancement_compute_errors(gt_enhanced, pred_enhanced)

            eval_measures[:num_metrics] += torch.tensor(measures).cuda(device=self.device)
            eval_measures[num_metrics] += 1

        if self.peeking_num == len(self.dataloader_eval.data):
            self.peeking_num = 0
        else:
            self.peeking_num = self.peeking_num + 1
        
        
        if opt.multiprocessing_distributed:
            group = dist.new_group([i for i in range(self.ngpus)])
            dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

        if not opt.multiprocessing_distributed or self.device == 0:
            eval_measures_cpu = eval_measures.cpu()
            cnt = eval_measures_cpu[num_metrics].item()
            eval_measures_cpu /= cnt
            print(space1+'🚀 E.H: Computing errors for {} eval samples'.format(int(cnt)))

            error_string = ''
            for i in range(num_metrics):
                error_string += '{}:{:.4f} '.format(self.eval_metrics[i], eval_measures_cpu[i])
            print(space1 + error_string)
        
        result = {'eval_measures': eval_measures_cpu, 'error_string': error_string}
        return result


    def check_best_eval_lower_better(self,
                                     metric, 
                                     eval_measures, 
                                     best_eval_measures_lower_better, 
                                     best_eval_steps, 
                                     global_step, 
                                     ):
        
        is_best = False
        if eval_measures < best_eval_measures_lower_better:
            old_best = best_eval_measures_lower_better.item()
            best_eval_measures_lower_better = eval_measures.item()
            is_best = True

        if is_best:
            old_best_step = best_eval_steps
            old_best_name = '/eh_model-{}-best_{}_{:.5f}.pth'.format(old_best_step, metric, old_best)
            model_path = self.checkpoint_dir + old_best_name
            if os.path.exists(model_path):
                command = 'rm {}'.format(model_path)
                os.system(command)
            best_eval_steps = global_step
            model_save_name = '/eh_model-{}-best_{}_{:.5f}.pth'.format(global_step, metric, eval_measures)
            print('E.H: New best for {}.'.format(model_save_name))
            
            result = {'best_eval_measures_lower_better':best_eval_measures_lower_better, 
                    'model_save_name': model_save_name, 
                    'best_eval_steps': best_eval_steps}
            return result
        else:
            result = None
            return result  


    def check_best_eval_higher_better(self,
                                      metric, 
                                      eval_measures, 
                                      best_eval_measures_higher_better, 
                                      best_eval_steps, 
                                      global_step
                                      ): 
    
        is_best = False
        if eval_measures > best_eval_measures_higher_better:
            old_best = best_eval_measures_higher_better.item()
            best_eval_measures_higher_better = eval_measures.item()
            is_best = True

        if is_best:
            old_best_step = best_eval_steps
            old_best_name = '/eh_model-{}-best_{}_{:.5f}.pth'.format(old_best_step, metric, old_best)
            model_path = self.checkpoint_dir + old_best_name
            if os.path.exists(model_path):
                command = 'rm {}'.format(model_path)
                os.system(command)
            
            best_eval_steps = global_step
            model_save_name = '/eh_model-{}-best_{}_{:.5f}.pth'.format(global_step, metric, eval_measures)
            print('E.H: New best for {}.'.format(model_save_name))
            
            result = {'best_eval_measures_higher_better':best_eval_measures_higher_better, 
                      'model_save_name':model_save_name, 
                      'best_eval_steps':best_eval_steps}
            return result
        else:
            result = None
            return result 
    
    
    def evalutate_worker(self, opt, model, global_step):
        
        result_commpute = self.enhanced_eval(opt, model)
        
        eval_measures = result_commpute['eval_measures']
        error_string = result_commpute['error_string']
        
        loss_list = []
        
        for idx in range(self.metrics_len):
            loss_list.append(eval_measures[idx])
            
            if idx < 3:
                result = self.check_best_eval_lower_better(self.eval_metrics[idx],
                                                           eval_measures[idx], 
                                                           self.best_eval_measures_lower_better[idx], 
                                                           self.best_eval_steps[idx], 
                                                           global_step
                                                           )
                if result != None:
                    self.best_eval_measures_lower_better[idx] = result['best_eval_measures_lower_better']
                    model_save_name = result['model_save_name']
                    self.best_eval_steps[idx] = result['best_eval_steps']
            
            elif idx >= 3:
                result = self.check_best_eval_higher_better(self.eval_metrics[idx],
                                                            eval_measures[idx], 
                                                            self.best_eval_measures_higher_better[idx-3],
                                                            self.best_eval_steps[idx], 
                                                            global_step
                                                            )
                if result != None:
                    self.best_eval_measures_higher_better[idx-3] = result['best_eval_measures_higher_better']
                    model_save_name = result['model_save_name']
                    self.best_eval_steps[idx] = result['best_eval_steps']   
                                                
                                                
            if result != None and self.is_checkpoint_save is True:
                if opt.distributed:
                    checkpoint = {'global_step': global_step,
                                    'model': model.module.state_dict(),
                                    'best_eval_measures_higher_better': self.best_eval_measures_higher_better,
                                    'best_eval_measures_lower_better': self.best_eval_measures_lower_better,
                                    'best_eval_steps': self.best_eval_steps
                    }
                else:
                    checkpoint = {'global_step': global_step,
                                    'model': model.state_dict(),
                                    'best_eval_measures_higher_better': self.best_eval_measures_higher_better,
                                    'best_eval_measures_lower_better': self.best_eval_measures_lower_better,
                                    'best_eval_steps': self.best_eval_steps
                    }
                torch.save(checkpoint, self.checkpoint_dir + model_save_name)
                print("Sucess to save '{}'.".format(model_save_name))
                
        result_commpute = {'eval_measures':loss_list, 
                           'val_sample': self.val_sample,
                           'val_image_tag_list': ['eh_val_origin', 'eh_val_est', 'eh_val_gt'],
                           'eval_metrics':self.eval_metrics, 
                           'error_string': error_string
                           }
        return result_commpute
