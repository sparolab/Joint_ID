

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import glob
from PIL import Image
import cv2
from tqdm import tqdm

image_path_list =  glob.glob('dataset_root/Dataset/My_UW/fake_images/diml_fake/test/*')
image_path_list.sort()

print('image_path_list: ', len(image_path_list))

meanRGB = []
stdRGB = []
for sample in tqdm(image_path_list[:7000]):
    mean = np.mean(np.array(Image.open(sample), dtype=np.float32)/255.0, axis=(0,1))
    meanRGB.append(mean)
    
    std = np.std(np.array(Image.open(sample), dtype=np.float32)/255.0, axis=(0,1))
    stdRGB.append(std)

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print(meanR, meanG, meanB)
print(stdR, stdG, stdB)

# mat_file_name = 