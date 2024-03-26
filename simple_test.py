import torch
from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv

a = torch.zeros((1, 3, 10, 10)).cuda()
b = torch.zeros((1, 8, 8, 3, 3)).cuda()

AdaptiveConv.apply(a, b)