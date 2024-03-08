"""
# > Modules for computing the Underwater Image Quality Measure (UIQM)
# Maintainer: Jahid (email: islam034@umn.edu)
"""
from scipy import ndimage
from PIL import Image
import numpy as np
import math
from skimage import color


def uciqe_compute(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    ## sc = (np.mean((chroma - uc)**2))**0.5
    sc = (np.mean(np.mean((chroma**2)-(uc**2))))**0.5

    ## #2nd term
    ## top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    ## sl = np.sort(l,axis=None)
    ## isl = sl[::-1]
    ## conl = np.mean(isl[:top])-np.mean(sl[:top])


    ## #3rd term
    ## satur = []
    ## chroma1 = chroma.flatten()
    ## l1 = l.flatten()
    ## for i in range(len(l1)):
    ##     if chroma1[i] == 0: satur.append(0)
    ##     elif l1[i] == 0: satur.append(0)
    ##     else: satur.append(chroma1[i] / l1[i])

    ## us = np.mean(satur)

    saturation = chroma / l
    u_s = np.mean(saturation)
    
    contrast_l = l.max() - l.min()
    
    uciqe = c1 * sc + c2 * contrast_l + c3 * u_s

    return uciqe


def rgb2lab_n(image):
    
    T = 0.008856
    height, width, ch =  image.shape
    N = height * height
    
    R = image[0].flatten()
    G = image[1].flatten()
    B = image[2].flatten()
    
    MAT = [[0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]]
    
    MAT = np.array(MAT)
    
    XYZ = MAT * np.array