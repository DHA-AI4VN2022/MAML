import torch
import torch.nn as nn
from torch.nn import init 
from torch.autograd import Variable

def Linear(args, output_size, bias, bias_initializer=None, device='cpu'):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias(default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    """
    
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.data.size(1) for a in args]
    for shape in shapes:
        total_arg_size += shape
    
    # Now the computation.
    weights = nn.Parameter(torch.FloatTensor(total_arg_size, output_size)).to(device)
    init.xavier_uniform_(weights)
    #weights = Variable(torch.zeros(total_arg_size, output_size))
    if len(args) == 1:
        args_ = args[0].to(device)
        res = torch.matmul(args_, weights)
    else:
        args_ = torch.cat(args,1).to(device)
        res = torch.matmul(args_, weights)
    if not bias:
        return res
    
    if bias_initializer is None:
        biases = Variable(torch.zeros(output_size)).to(device)
        
    return torch.add(res, biases)