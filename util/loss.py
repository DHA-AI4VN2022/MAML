import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import torch

DEVICE = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

def mape_loss(pred, target, device, reduction='mean'):
    """
    input, output: tensor of same shape
    """
    target = torch.where(
        target == 0, 
        torch.tensor(1e-6), 
        target
    )
    diff = (pred - target) / target
    if reduction == 'mean':
        mape = diff.abs().mean()
    elif reduction == 'sum':
        mape = diff.abs().sum()
    else:
        mape = diff
    return mape

def r2_loss(pred, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2