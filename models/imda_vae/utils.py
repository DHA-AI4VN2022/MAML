import torch
import torch.nn as nn

def loss_function(x, x_hat, mean, log_var, config):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') / config['batch_size']
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD
