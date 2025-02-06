import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from fused_ssim import fused_ssim


C1 = 0.01 ** 2
C2 = 0.03 ** 2

    
def fast_ssim(img1, img2):
    ssim_map = fused_ssim(C1, C2, img1, img2)
    return ssim_map.mean()