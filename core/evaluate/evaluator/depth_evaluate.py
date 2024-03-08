
import torch
import tqdm
import numpy as np
import torch.distributed as dist
import os

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

def depth_compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def depth_eval(opt, model, dataloader_eval, gpu, ngpus):
    space1 = " "*5
    
    num_metrics = len(eval_metrics)
    
    if gpu != None:
        eval_measures = torch.zeros(num_metrics + 1).cuda(device=gpu)
    else:
        eval_measures = torch.zeros(num_metrics + 1)
        
    for _, eval_sample_batched in tqdm.tqdm(enumerate(dataloader_eval.data), 
                                            total=len(dataloader_eval.data)) if opt.rank == 0 or not opt.multiprocessing_distributed else enumerate(dataloader_eval.data):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))

            gt_depth = eval_sample_batched['depth']

            _, _, _, _, pred_depth = model(image)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if opt.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < opt.min_depth_eval] = opt.min_depth_eval
        pred_depth[pred_depth > opt.max_depth_eval] = opt.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = opt.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = opt.min_depth_eval

        valid_mask = np.logical_and(gt_depth > opt.min_depth_eval, gt_depth < opt.max_depth_eval)

        measures = depth_compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:num_metrics] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[num_metrics] += 1

    if opt.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not opt.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))

        error_string = ''
        for i in range(num_metrics):
            error_string += '{}:{:.4f} '.format(eval_metrics[i], eval_measures_cpu[i])
        print(space1 + error_string)
        result = {'eval_measures': eval_measures_cpu, 'eval_metrics': eval_metrics, 'error_string': error_string}
        return result
    else:
        return None

# -> {save_path: save_model_name: checkpoint}
def check_best_eval_lower_better(metric, eval_measures, best_eval_measures_lower_better, best_eval_steps, global_step, save_dir): 
    is_best = False
    if eval_measures < best_eval_measures_lower_better:
        old_best = best_eval_measures_lower_better.item()
        best_eval_measures_lower_better = eval_measures.item()
        is_best = True

    if is_best:
        old_best_step = best_eval_steps
        old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, metric, old_best)
        model_path = save_dir + old_best_name
        if os.path.exists(model_path):
            command = 'rm {}'.format(model_path)
            os.system(command)
        best_eval_steps = global_step
        model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, metric, eval_measures)
        print('New best for {}. Saving model: {}'.format(eval_measures, model_save_name))
        
        result = {'best_eval_measures_lower_better':best_eval_measures_lower_better, 
                  'model_save_name': model_save_name, 
                  'best_eval_steps': best_eval_steps}
        return result
    else:
        result = None
        return result  


def check_best_eval_higher_better(metric, eval_measures, best_eval_measures_higher_better, best_eval_steps, global_step, save_dir): 
    is_best = False
    if eval_measures > best_eval_measures_higher_better:
        old_best = best_eval_measures_higher_better.item()
        best_eval_measures_higher_better = eval_measures.item()
        is_best = True

    if is_best:
        old_best_step = best_eval_steps
        old_best_name = 'model-{}-best_{}_{:.5f}'.format(old_best_step, metric, old_best)
        model_path = save_dir + old_best_name
        if os.path.exists(model_path):
            command = 'rm {}'.format(model_path)
            os.system(command)
        best_eval_steps = global_step
        model_save_name = 'model-{}-best_{}_{:.5f}'.format(global_step, metric, eval_measures)
        print('New best for {}. Saving model: {}'.format(eval_measures, model_save_name))
        
        result = {'best_eval_measures_higher_better':best_eval_measures_higher_better, 
                  'model_save_name': model_save_name, 
                  'best_eval_steps': best_eval_steps}
        return result
    else:
        result = None
        return result  
    
