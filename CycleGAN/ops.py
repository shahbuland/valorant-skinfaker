import torch
from torch import nn
import torch.nn.functional as F
from .constants import *
import numpy as np

# Fixes T wrt data type
def Tensor(T):
	T = T.float()
	if USE_CUDA: T = T.cuda()
	return T

# Gets np image from model tensor
def npimage(A):
	A = A.detach().cpu().numpy()
	A = np.moveaxis(A,1,3)	
	return A

# Gets labels for discriminator
def get_labels(val,size):
	return Tensor(val*torch.ones(size,1,4,4))

# Return value t places a and b
def lerp(a, b, t):
        return a + t*(b - a)

# LR change with epoch
def lr_map(lr, epoch):
        if(epoch < 100):
                return lr
        elif(epoch >= 200):
                return 0
        else:
                return lerp(lr, 0, (lr - 100)/100)
