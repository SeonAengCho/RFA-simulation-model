import torch
import torch.nn as nn
import numpy as np

def diceloss(pred, target):
    
    smooth = 1.
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - (2 * (intersection + smooth) / (A_sum + B_sum + smooth)) 
   
