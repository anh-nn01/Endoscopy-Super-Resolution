"""
Generate a Super Resolution (HR-LR) image dataset as numerical dataset
Format: [ [HR_0, LR_0], [HR_1, LR_1], ... , [HR_n, LR_n] ]
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

from LR_gen import Generate_LR

"""
save_file_dir: the directory to save generated data file
img_dir: the directory to dataset of HR patches
scale_factor: down-scaling factor, can be "2", "4", or "8"
"""
def Generate_Dataset(img_dir : str, scale_factor : int):
    SR_dataset = []
    data_file_name = "SRdataset_x" + str(scale_factor) + ".pickle"

    for img_name in os.listdir(img_dir):
        HR = plt.imread(os.path.join(img_dir, img_name))
        LR = Generate_LR(HR, scale_factor = scale_factor)
        SR_dataset.append( [ HR, LR ] )
        
    with open(os.path.join("/Users/nhunh/SRMed", data_file_name), "wb") as file:
        pkl.dump(SR_dataset, file)

if(__name__ == "__main__"):
    Generate_Dataset(img_dir = "/Users/nhunh/SRMed/Dataset-Crop-Image/Train-HR/", scale_factor = 2)
