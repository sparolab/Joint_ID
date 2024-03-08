

from cv2 import transform
import numpy as np
from sklearn.utils import resample
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms as tr
from PIL import Image
import os
import random
from ..dataload_builder import DATALOADER


@DATALOADER.register_module()
class Joint_De_Eh_Preprocess(Dataset):
    def __init__(self,
                 mode,
                 auto_crop,
                 dataset_txt_file,
                 img_size,
                 data_path,
                 depth_gt_path,
                 enhanced_gt_path,
                 argumentation=None,
                 is_save_gt_image=False,
                 ):
        
        if argumentation is None and mode is 'train':
            raise ValueError("If 'mode' is 'train, then 'argumentation' must be not None")
        
        if not os.path.isfile(dataset_txt_file) and dataset_txt_file != '':
            raise FileExistsError("{} file is not exist. Please check your file path".format(dataset_txt_file))
        with open(dataset_txt_file, 'r') as f:
            self.filenames = f.readlines()
            print("len(train.filenames): ", len(self.filenames))

        self.mode = mode
        self.transform = self.preprocessing_transforms(mode, is_save_gt_image)
        
        self.auto_crop = auto_crop
        self.argumentation = argumentation
        self.img_size = img_size
        self.data_path = data_path
        self.depth_gt_path = depth_gt_path
        self.enhanced_gt_path = enhanced_gt_path
        
        self.is_save_gt_image = is_save_gt_image
    
    
    def __getitem__(self, index):
        sample_path = self.filenames[index]

        if self.mode == 'train':
            image_path = sample_path.split()[0]
            enhanced_gt =  sample_path.split()[1]
            depth_gt = sample_path.split()[2]
            depth_scaling = float(sample_path.split()[3])
            
            sample = self.train_dataloader(image_path, enhanced_gt, depth_gt, depth_scaling)
            
        elif self.mode == 'test': 
            image_path = sample_path.split()[0]
            image = Image.open(os.path.join(self.data_path, image_path))
            
            if self.is_save_gt_image is True:
                enhanced_gt =  sample_path.split()[1]
                depth_gt = sample_path.split()[2]
                depth_scaling = float(sample_path.split()[3])

                depth_gt = Image.open(os.path.join(self.depth_gt_path, depth_gt))
                enhanced_gt = Image.open(os.path.join(self.enhanced_gt_path, enhanced_gt))

            if self.auto_crop is True:
                auto_height = 32 * (image.height // 32)
                auto_width = 32 * (image.width // 32)   
                top_margin = int((image.height - auto_height) / 2) 
                left_margin = int((image.width - auto_width) / 2)             

                image = image.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height)) 
                
                if self.is_save_gt_image is True:
                    depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height))  
                    enhanced_gt = enhanced_gt.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height)) 

            else:
                if self.argumentation['do_resize_crop'] is True:
                    auto_height = 32 * (self.img_size[0] // 32)
                    auto_width = 32 * (self.img_size[1] // 32)  
                    
                    image = image.resize((auto_width, auto_height), Image.BICUBIC)
                    
                    if self.is_save_gt_image is True:
                        enhanced_gt = enhanced_gt.resize((auto_width, auto_height), Image.BICUBIC)                
                        depth_gt = depth_gt.resize((auto_width, auto_height), Image.NEAREST)
                
                elif self.argumentation['do_center_crop'] is True:
                    height = image.height
                    width = image.width
                    top_margin = int((height - self.img_size[0]) / 2) 
                    left_margin = int((width - self.img_size[1]) / 2)
                    
                    image = image.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))  
                    
                    if self.is_save_gt_image is True:
                        depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))
                        enhanced_gt = enhanced_gt.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))
            
            if self.is_save_gt_image is True:    
                image = np.asarray(image, dtype= np.float32) / 255.0
                enhanced_gt = np.asarray(enhanced_gt, dtype= np.float32) / 255.0
                
                depth_gt = self.rotate_image(depth_gt, 0, flag= Image.NEAREST)
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / depth_scaling
                depth_gt = np.expand_dims(depth_gt, axis= 2)
                depth_gt[depth_gt > 600] = 0
                
                sample = {'image': image, 'depth': depth_gt, 'enhanced': enhanced_gt}
            else:
                image = np.asarray(image, dtype= np.float32) / 255.0
                sample = {'image': image}
            
            
        elif self.mode == 'eval':
            image_path = sample_path.split()[0]
            enhanced_gt =  sample_path.split()[1]
            depth_gt = sample_path.split()[2]
            depth_scaling = float(sample_path.split()[3])

            image = Image.open(os.path.join(self.data_path, image_path))
            depth_gt = Image.open(os.path.join(self.depth_gt_path, depth_gt))
            enhanced_gt = Image.open(os.path.join(self.enhanced_gt_path, enhanced_gt))

            if self.auto_crop is True:
                auto_height = 32 * (image.height // 32)
                auto_width = 32 * (image.width // 32)   
                top_margin = int((image.height - auto_height) / 2) 
                left_margin = int((image.width - auto_width) / 2)             

                image = image.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height)) 
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height))  
                enhanced_gt = enhanced_gt.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height))  

            else:
                if self.argumentation['do_resize_crop'] is True:
                    auto_height = 32 * (self.img_size[0] // 32)
                    auto_width = 32 * (self.img_size[1] // 32)  

                    image = image.resize((auto_width, auto_height), Image.BICUBIC)
                    enhanced_gt = enhanced_gt.resize((auto_width, auto_height), Image.BICUBIC)
                    depth_gt = depth_gt.resize((auto_width, auto_height), Image.NEAREST)
                    
                elif self.argumentation['do_center_crop'] is True:
                    height = image.height
                    width = image.width
                    top_margin = int((height - self.img_size[0]) / 2) 
                    left_margin = int((width - self.img_size[1]) / 2)
                    
                    image = image.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))  
                    depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))
                    enhanced_gt = enhanced_gt.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))                
                
            image = np.asarray(image, dtype= np.float32) / 255.0
            enhanced_gt = np.asarray(enhanced_gt, dtype= np.float32) / 255.0
            
            depth_gt = self.rotate_image(depth_gt, 0, flag= Image.NEAREST)
            depth_gt = np.asarray(depth_gt, dtype=np.float32) / depth_scaling
            depth_gt = np.expand_dims(depth_gt, axis= 2)
            depth_gt[depth_gt > 600] = 0
                        
            sample = {'image': image, 'depth': depth_gt, 'enhanced': enhanced_gt}
        else:
            raise ValueError("'self.mode' is must be ['train','eval','test']. But, Got {}".format(self.mode))
             
        if self.transform:
            sample = self.transform(sample)
        
        return sample


    def train_dataloader(self, input_path, enhanced_gt, depth_gt, depth_scaling):
        image = Image.open(os.path.join(self.data_path, input_path))
        enhanced_gt = Image.open(os.path.join(self.enhanced_gt_path, enhanced_gt))
        depth_gt = Image.open(os.path.join(self.depth_gt_path, depth_gt))
        
        if self.auto_crop is True:
            auto_height = 32 * (image.height // 32)
            auto_width = 32 * (image.width // 32)   
            top_margin = int((image.height - auto_height) / 2) 
            left_margin = int((image.width - auto_width) / 2)             

            image = image.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height)) 
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height))  
            enhanced_gt = enhanced_gt.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height))  
        
        else:
            if self.argumentation['do_resize_crop'] is True:
                auto_height = 32 * (self.img_size[0] // 32)
                auto_width = 32 * (self.img_size[1] // 32)  

                image = image.resize((auto_width, auto_height), Image.BICUBIC)
                enhanced_gt = enhanced_gt.resize((auto_width, auto_height), Image.BICUBIC)                
                depth_gt = depth_gt.resize((auto_width, auto_height), Image.NEAREST)
            
            elif self.argumentation['do_center_crop'] is True:
                height = image.height
                width = image.width

                top_margin = int((height - self.img_size[0]) / 2) 
                left_margin = int((width - self.img_size[1]) / 2)
    
                if self.argumentation['do_random_crop'] is True:
                    top_margin = random.randint(0, top_margin*2)
                    left_margin = random.randint(0, left_margin*2)

                image = image.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))   
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))
                enhanced_gt = enhanced_gt.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))
                       
        if self.argumentation['do_random_rotate'] is True:
            random_angle = (random.random() - 0.5) * 2 * self.argumentation['degree']
            image = self.rotate_image(image, random_angle, Image.BICUBIC)
            enhanced_gt = self.rotate_image(enhanced_gt, random_angle, Image.BICUBIC)
            depth_gt = self.rotate_image(depth_gt, random_angle, flag= Image.NEAREST)
        else:
            depth_gt = self.rotate_image(depth_gt, 0, flag= Image.NEAREST)
        
        image = np.asarray(image, dtype=np.float32) / 255.0
        enhanced_gt = np.asarray(enhanced_gt, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32) / depth_scaling
        depth_gt = np.expand_dims(depth_gt, axis=2)
    
        # dataset argumentation
        if self.argumentation['do_horison_flip'] is True:
            image, depth_gt, enhanced_gt = self.random_flip(image, depth_gt, enhanced_gt)

        if self.argumentation['do_augment_color'] is True:
            image = self.augment_image(image)
        
        depth_gt[depth_gt > 600] = 0
        
        sample = {'image': image, 'depth': depth_gt, 'enhanced': enhanced_gt}

        return sample

    # About preprocessing functions
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result
    
    def random_crop(self, image, depth, height, width):
    
        assert image.shape[0] >= height,            "image.shape[0] < height"
        assert image.shape[1] >= width,             "image.shape[1] < width"
        assert depth.shape[0] == image.shape[0],    "depth.shape[0] != image.shape[0]"
        assert depth.shape[1] == image.shape[1],    "depth.shape[1] != image.shape[1]"

        x = random.randint(0, image.shape[1] - width)
        y = random.randint(0, image.shape[0] - height)
        image = image[y:y+height, x:x+width, :]
        depth = depth[y:y+height, x:x+width, :]

        return image, depth

    def random_flip(self, image, depth_gt, enhanced_gt):
        do_flip = random.random()
        
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            enhanced_gt = (enhanced_gt[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        return image, depth_gt, enhanced_gt  
    
    def augment_image(self, image):
        do_augment = random.random()
        
        if do_augment > 0.3:
            # gamma augmentation
            gamma = random.uniform(0.9, 1.1)
            image_aug = image ** gamma
            
            # brightness augmentation
            brightness = random.uniform(0.5, 1.1)
            image_aug = image * brightness

            # color augmentation
            colors = np.random.uniform(0.9, 1.1, size=3)    # size=3 -> channels: 3 이라서
            white = np.ones((image.shape[0], image.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)

            image_aug = image_aug * color_image
            image_aug = np.clip(image_aug, 0, 1)

            return image_aug
        else:
            return image
        
    
    def __len__(self):
        return len(self.filenames)

    def preprocessing_transforms(self, mode, is_save_gt_image):
        return tr.Compose([To_myTensor(mode= mode, is_save_gt_image= is_save_gt_image)])


class To_myTensor(object):
    def __init__(self, mode: str, is_save_gt_image: bool):
        self.is_save_gt_image = is_save_gt_image
        self.mode = mode
        self.normalize = tr.Normalize(mean=[0.13553666, 0.41034216, 0.34636855], std=[0.04927989, 0.10722694, 0.10722694])
    
    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test' and self.is_save_gt_image is False:
            return {'image': image}
        else:
            depth_gt = sample['depth']
            depth_gt = self.to_tensor(depth_gt)
            
            enhanced_gt = sample['enhanced']
            enhanced_gt = self.to_tensor(enhanced_gt)
            
            return {'image': image, 'depth': depth_gt, 'enhanced': enhanced_gt}


    def to_tensor(self, img):
        if not (self._is_pil_image(img) or self._is_numpy_image(img)):
            raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(img)))
        
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose((2, 0 ,1))).float()
            return img
    
    def _is_pil_image(self, img):
        return isinstance(img, Image.Image)
    
    def _is_numpy_image(self, img: np.ndarray):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})