"""
Down Sampling for dataset creation
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import cv2

"""
Create Low Resolution image for LR dataset
param scale factor: scale to be downsample
"""
def Generate_LR(image, scale_factor):
    # Gaussian Blur
    LR = cv2.GaussianBlur(image, ksize=(5,5), sigmaX=5, sigmaY=5)

    # Bicubic Rescaling
    LR = cv2.resize(LR, (LR.shape[1]//scale_factor, LR.shape[0]//scale_factor), interpolation = cv2.INTER_CUBIC)
    
    # Add Gaussian Noise
    LR = Gaussian_Noise(LR)
    
    return np.uint8(LR)

def Gaussian_Noise(image):
    mean = 10
    var = 10
    std = var**0.5
    
    gauss = np.random.normal(mean, std, image.shape)
    image = image + gauss
    return np.clip(image, 0, 255)


scale_factor = 2
HR = plt.imread("/Users/nhunh/SRMed/Dataset-Crop/Train-HR/train4.jpg")
Image.fromarray(HR).save("/Users/nhunh/Downloads/HR.jpg")

LR = Generate_LR(HR, scale_factor)
Image.fromarray(LR).save("/Users/nhunh/Downloads/LR.jpg")

SR = cv2.resize(LR, (LR.shape[1] * scale_factor, LR.shape[0] * scale_factor), interpolation = cv2.INTER_CUBIC)
Image.fromarray(SR).save("/Users/nhunh/Downloads/SR.jpg")

