
import torch
import tqdm
import numpy as np
import torch.distributed as dist
import os
import math

eval_metrics = ['abs_rel', 'rms', 'sq_rel', 'log_rms', 'psnr']

def enhancement_compute_errors(gt, pred):
    mse = np.mean( (gt - pred) ** 2 )
    psnr = 10 * math.log10(255.0**2/mse)
    
    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return [abs_rel, rms, sq_rel, log_rms, psnr]


def enhancement_eval(opt, model, dataloader_eval, gpu, ngpus):
    space1 = " "*5
    
    num_metrics = len(eval_metrics)
    
    if gpu != None:
        eval_measures = torch.zeros(num_metrics + 1).cuda(device=gpu)
    else:
        eval_measures = torch.zeros(num_metrics + 1)     
    for _, eval_sample_batched in tqdm.tqdm(enumerate(dataloader_eval.data), 
                                            total=len(dataloader_eval.data)) if opt.rank == 0 or not opt.multiprocessing_distributed else enumerate(dataloader_eval.data):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['underwater_image'].cuda(gpu, non_blocking=True))

            gt_clean_image = eval_sample_batched['enhanced_image']
            pred_clean_image = model(image)

            pred_clean_image = pred_clean_image.cpu().numpy().squeeze()
            gt_clean_image = gt_clean_image.cpu().numpy().squeeze()

        if opt.do_kb_crop:
            height, width = gt_clean_image.shape
            top_margin = int(height - 0)
            left_margin = int(width - 0)
            pred_clean_image_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_clean_image_uncropped[top_margin:top_margin + 0, left_margin:left_margin + 0] = pred_clean_image
            pred_clean_image = pred_clean_image_uncropped

        total_psnr_error = enhancement_compute_errors(pred_clean_image, gt_clean_image)
        eval_measures[:num_metrics] += torch.tensor(total_psnr_error).cuda(device=gpu)

        eval_measures[num_metrics] += 1

    if opt.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not opt.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[num_metrics].item()
        eval_measures_cpu /= cnt
        print(space1+'🚀 Computing errors for {} eval samples'.format(int(cnt)))
        
        error_string = ''
        for i in range(num_metrics):
            error_string += '{}:{:.4f} '.format(eval_metrics[i], eval_measures_cpu[i])
        print(space1 + error_string)
        result = {'eval_measures': eval_measures_cpu, 'eval_metrics': eval_metrics, 'error_string': error_string}
        return result
    else:
        return None


# -> {save_path: save_model_name: checkpoint}
def check_best_eval_lower_better(idx, eval_measures, best_eval_measures_lower_better, best_eval_steps, global_step, save_dir): 
    space1 = " "*5
    
    is_best = False
    if eval_measures < best_eval_measures_lower_better:
        print("eval_measures:", eval_measures)
        print("best_eval_measures_lower_better:", best_eval_measures_lower_better)
        old_best = best_eval_measures_lower_better.item()
        best_eval_measures_lower_better = eval_measures.item()
        is_best = True

    if is_best == True:
        old_best_step = best_eval_steps
        old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[idx], old_best)
        model_path = save_dir + '/' + old_best_name
        if os.path.exists(model_path):
            command = 'rm {}'.format(model_path)
            os.system(command)
        best_eval_steps = global_step
        model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[idx], eval_measures)
        print(space1+'New best for {}. Saving model: {}'.format(eval_measures, model_save_name))
        
        result = {'best_eval_measures_lower_better':best_eval_measures_lower_better, 
                  'model_save_name': model_save_name, 
                  'best_eval_steps': best_eval_steps,
                  'old_best_step': old_best_step}
        return result
    else:
        result = None
        return result  


def check_best_eval_higher_better(idx, eval_measures, best_eval_measures_higher_better, best_eval_steps, global_step, save_dir): 
    space1 = " "*5
    
    is_best = False
    if eval_measures > best_eval_measures_higher_better:
        print("eval_measures:", eval_measures)
        print("best_eval_measures_higher_better:", best_eval_measures_higher_better)
        old_best = best_eval_measures_higher_better.item()
        best_eval_measures_higher_better = eval_measures.item()
        is_best = True

    if is_best == True:
        old_best_step = best_eval_steps
        old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[idx], old_best)
        model_path = save_dir + '/' + old_best_name
        if os.path.exists(model_path):
            command = 'rm {}'.format(model_path)
            os.system(command)
        best_eval_steps = global_step
        model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[idx], eval_measures)
        print(space1+'New best for {}. Saving model: {}'.format(eval_measures, model_save_name))
        
        result = {'best_eval_measures_higher_better':best_eval_measures_higher_better, 
                  'model_save_name': model_save_name, 
                  'best_eval_steps': best_eval_steps,
                  'old_best_step': old_best_step}
        return result
    else:
        result = None
        return result  
    
