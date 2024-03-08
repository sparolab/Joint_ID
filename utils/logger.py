
from tensorboardX import SummaryWriter
import torchvision
import os
import wandb
from pathlib import Path
from PIL import Image
import numpy as np
import torch


FILE = Path(__file__).resolve()                     # 파일의 절대 경로
ROOT = FILE.parents[1]                              # 파일의 상위 폴더
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))      # 워크스페이스에서의 상대 파일 경로 
SAVE_ROOT = os.path.join(ROOT, 'save/summaries')


class TensorBoardLogger(object): 
    """Tensorboard logger."""
 
    def __init__(self, log_dir, comment, flush_sec: int = None):
        """Initialize summary writer."""
        # self.writer = tf.summary.FileWriter(log_dir)
        if flush_sec != None:
            self.writer = SummaryWriter(comment=comment, log_dir = log_dir)
        else:
            self.writer = SummaryWriter(comment=comment, log_dir = log_dir, flush_secs= flush_sec)
        
    def scalar_summary(self, tag_list: list, value_list: list, step: int):
        """Add scalar."""
        for tmp_value, tmp_tag in zip(value_list, tag_list):
            self.writer.add_scalar(tmp_tag, tmp_value, step)
    
    def model_graph_summary(self, model, network_input):
        self.writer.add_graph(model, network_input, verbose=False)
    
    def image_summary(self, tag_list: list, image_list: list, step: int, data_format = 'CHW'):
        for tmp_image, tmp_tag in zip(image_list, tag_list): 
            # img_grid = torchvision.utils.make_grid(torch.Tensor(tmp_image))
            self.writer.add_image(tmp_tag, tmp_image, step, dataformats=data_format)
    
    def log_flush(self):
        self.writer.flush()

        
class Wandb_Logger():
    def __init__(self, opt, ngpus_per_node, param, logger_name):
        config = dict(project=opt.project_name,
                    device= ("gpu_nums: {}".format(ngpus_per_node) if ngpus_per_node >= 1 else "cpu"),
                    epochs= opt.num_epochs,
                    batch_size=opt.batch_size * ngpus_per_node,
                    learning_rate= opt.learning_rate
        )  
        self.wandb = wandb
        self.wandb.login()            
            
        if opt.wandb_save_path == '':
            self.wandb.init(project=opt.project_name, name=logger_name, config=param)   
            
        else:
            if os.path.isdir(opt.wandb_save_path):
                self.save_path = os.path.join(opt.wandb_save_path, opt.log_comment)  
                try:
                    os.mkdir(self.save_path)
                except:
                    print(f"File exists: '{self.save_path}'")
                self.wandb.init(project=opt.project_name, name=logger_name, dir= self.save_path, config=param)    
            else:
                raise IsADirectoryError("There is not the directory. Got {}".format(os.wandb_save_path))
                   
    def logging_save(self, global_step):
        save_file_path = os.path.join(self.save_path, f"{global_step}.h5")
        self.wandb.save(save_file_path)
        print("The wandb model is saved in '{}'".format(save_file_path))

    def logging_table(self, total_tag:str, images:list, image_tag:list, predicted:list, labels:list, probs:list = None, loading_step: int = None):
        length = len(probs[0])
        table = self.wandb.Table(columns=["image", "pred", "labels"]+[f"probs{i}" for i in range(length)])
        
        for img, tag, pred, label, prob in zip(images, image_tag, predicted, labels, probs):
            tmp_img = self.wandb.Image(img, caption= tag)
            table.add_data(tmp_img, pred, label, *prob)
        
        self.wandb.log({total_tag:table}, step=loading_step)

    def logging_model_watch(self, model, criterion, log_freq, model_idx:int=None):
        self.wandb.watch(model, criterion, log='all', log_freq=log_freq, idx=model_idx)
    
    def logging_images(self, total_tag:str, images:list, tag_list:list, loading_step: int, group_idx: int = None):
        if (isinstance(images, list) and isinstance(tag_list, list)):
            if not(isinstance(images[0], np.ndarray) or isinstance(images[0], Image.Image)):
                raise TypeError("Type of the images must be np.ndarray or PIL.Image. but Got {}".format(type(images[0])))
            
            log_list = []
            for tmp_image, tmp_tag in zip(images, tag_list):
                wandb_image = self.wandb.Image(tmp_image, caption= tmp_tag, grouping= group_idx)
                log_list.append(wandb_image)                

            self.wandb.log({total_tag: log_list}, step= loading_step)  
        else:
            raise TypeError("Type of the images must list or np.ndarray PIL.Image. but Got {}".format(type(images[0])))
    
    def logging_histogram(self, total_tag: str, metrics, loading_step: int):
        self.wandb.log({total_tag: wandb.Histogram(metrics)}, step= loading_step)
              
    def logging_graph(self, graph_tag: list, graph_value: list, loading_step: int):
        if (isinstance(graph_tag, list) and isinstance(graph_value, list)):
    
            for tmp_tag, tmp_value in zip(graph_tag, graph_value):
                self.wandb.log({tmp_tag: tmp_value}, step= loading_step)
        else:
            raise TypeError("The graph_tag and graph_value must be list. but Got {}, {}".format(type(graph_tag), type(graph_value)))
    
    def logging_video(self, total_tag: str, video_path: str, fps: int, video_format:str = None):
        if video_format == None or not video_format in ['gif','mp4','webm','ogg']:
            raise ValueError("The format must be 'gif', 'mp4', 'webm' or 'ogg'. But, Got {}".format(video_format))
        
        video = self.wandb.Video(video_path, fps=fps, format=video_format)
        self.wandb.log({total_tag:video})
    
    def logging_alert(self, title: str, text: str):
        self.wandb.alert(title = title, text = text)
        print("🔔🔔 Alarm is happend 🔔🔔")
    
    def logging_train_ended(self, contents: dict = None):        
        if contents != None:
            self.wandb.summary.update(contents)
        self.wandb.finish()
    
    def logging_segmetation(self, total_tag:str, input_tag:list, input_image: list, pred_masks: list, gt_masks: list, labels: dict, loading_step: int):
        if not isinstance(labels, dict):
            raise TypeError("Type of the labels must be 'tuple'. but Got {}".format(type(labels)))
        
        if (isinstance(input_image, list) and isinstance(input_tag, list)) and (isinstance(pred_masks, list) and isinstance(gt_masks, list)):
            if not(isinstance(input_image[0], np.ndarray) or isinstance(input_image[0], Image.Image)):
                raise TypeError("Type of the input_image must be np.ndarray or PIL.Image. but Got {}".format(type(input_image[0])))
            if not (isinstance(pred_masks[0], np.ndarray) and isinstance(gt_masks[0], np.ndarray)):
                raise TypeError("Type of the pred_masks, gt_masks must be np.ndarray or PIL.Image. but Got {} / {}".format(type(pred_masks[0]), 
                                                                                                                           type(gt_masks[0])))            
            log_list = []
            for tmp_image, tmp_tag, tmp_pred, tmp_gt in zip(input_image, input_tag, pred_masks, gt_masks):
                wandb_image = self.wandb.Image(tmp_image, caption= tmp_tag, masks={
                    "prediction" : {"mask_data": tmp_pred, "class_labels" : labels},
                    "ground truth" : {"mask_data": tmp_gt, "class_labels" : labels}
                })
                log_list.append(wandb_image)                

            self.wandb.log({total_tag: log_list}, step=loading_step)  
        else:
            raise TypeError("Type of the input_image, input_tag, pred_masks, gt_masks must be list. but Got {} / {} / {}".format(type(input_image),
                                                                                                                                 type(pred_masks),
                                                                                                                                 type(gt_masks)))

    def logging_object_detection(self, 
                            total_tag:str, 
                            input_tag:list, 
                            input_image: list, 
                            boxes_per_pred: list,
                            targets_per_pred: list,
                            scores_per_pred: list,
                            boxes_per_gt: list,
                            targets_per_gt: list,
                            labels: dict,
                            loading_step: int
        ):
        
        if not isinstance(labels, dict):
            raise TypeError("Type of the labels must be 'tuple'. but Got {}".format(type(labels)))
        
        if (isinstance(input_image, list) and isinstance(input_tag, list)) and (isinstance(boxes_per_pred, list) and isinstance(boxes_per_gt, list)):
            if not(isinstance(input_image[0], np.ndarray) or isinstance(input_image[0], Image.Image)):
                raise TypeError("Type of the input_image must be np.ndarray or PIL.Image. but Got {}".format(type(input_image[0])))
            if not (isinstance(boxes_per_pred[0], np.ndarray) and isinstance(boxes_per_gt[0], np.ndarray)):
                raise TypeError("Type of the pred_masks, gt_masks must be np.ndarray or PIL.Image. but Got {} / {}".format(type(boxes_per_pred[0]), 
                                                                                                                           type(boxes_per_gt[0])))            
            log_list = []
            for image_idx, tmp_image in enumerate(input_image):
                pred_boxes_per_image = []
                gt_boxes_per_image = []
                
                for idx, pred_box, pred_target, pred_score, gt_box, gt_target in enumerate(zip(boxes_per_pred[image_idx], 
                                                                                    targets_per_pred[image_idx], 
                                                                                    scores_per_pred[image_idx], 
                                                                                    boxes_per_gt[image_idx],
                                                                                    targets_per_gt[image_idx])):
                    pred_box_data = {"position" : {
                        "minX" : pred_box.xmin,
                        "maxX" : pred_box.xmax,
                        "minY" : pred_box.ymin,
                        "maxY" : pred_box.ymax},
                        "class_id" : pred_target,
                        # optionally caption each box with its class and score
                        "box_caption" : "%s (%.3f)" % (labels[pred_target], pred_score),
                        "domain" : "pixel",
                        "scores" : { "score" : pred_score }} 
                    pred_boxes_per_image.append(pred_box_data)
                      
                    gt_box_data = {"position" : {
                        "minX" : gt_box.xmin,
                        "maxX" : gt_box.xmax,
                        "minY" : gt_box.ymin,
                        "maxY" : gt_box.ymax},
                        "class_id" : gt_target,
                        # optionally caption each box with its class and score
                        "box_caption" : "%s" % (labels[gt_target]),
                        "domain" : "pixel",
                        "scores" : { "score" : 100.0 }}
                    gt_boxes_per_image.append(gt_box_data)
                    
                wandb_image = self.wandb.Image(tmp_image, caption= input_tag[image_idx], 
                                               boxes={
                                                   "prediction" : {"box_data": pred_boxes_per_image, "class_labels" : labels},
                                                   "ground truth" : {"box_data": gt_boxes_per_image, "class_labels" : labels}})
                log_list.append(wandb_image)
                              
            self.wandb.log({total_tag: log_list}, step=loading_step)  
        else:
            raise TypeError("Type of the input_image, input_tag, pred_masks, gt_masks must be list. but Got {} / {} / {}".format(type(input_image),
                                                                                                                                 type(boxes_per_pred),
                                                                                                                                 type(boxes_per_gt)))