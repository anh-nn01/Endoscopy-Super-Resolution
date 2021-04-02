import torch
import torch.nn as nn

def L1():
    return nn.L1Loss()

def L2():
    return nn.MSELoss()

def CrossEntropy():
    return nn.BCELoss()
