"""
Down Sampling for dataset creation
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import cv2
import os

"""
Create Low Resolution image for LR dataset
param scale factor: scale to be downsample
"""
def Generate_LR(image, scale_factor):
    # Gaussian Blur
    LR = cv2.GaussianBlur(image, ksize=(5,5), sigmaX=10, sigmaY=10)

    # Bicubic Rescaling
    LR = cv2.resize(LR, (LR.shape[1]//scale_factor, LR.shape[0]//scale_factor), interpolation = cv2.INTER_CUBIC)
    
    # Add Gaussian Noise
    LR = Gaussian_Noise(LR)
    
    return np.uint8(LR)

def Gaussian_Noise(image):
    mean = 15
    var = 15
    std = var**0.5
    
    gauss = np.random.normal(mean, std, image.shape)
    image = image + gauss
    return np.clip(image, 0, 255)

if(__name__ == "__main__"):
    for img_name in os.listdir("/Users/nhunh/SRMed/Dataset-Crop-Image/Train-HR/",):
        HR = plt.imread(os.path.join("/Users/nhunh/SRMed/Dataset-Crop-Image/Train-HR/", img_name))
        LR = Generate_LR(HR, scale_factor = 2)
        Image.fromarray(LR).save(os.path.join("/Users/nhunh/SRMed/Dataset-Crop-Image/Train-LR/", "LRx2_" + img_name))

