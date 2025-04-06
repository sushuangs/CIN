import torch.nn as nn
import random
import numpy as np
import torch


#
class GaussianNoise(nn.Module):
    def __init__(self, opt, device):
        super(GaussianNoise, self).__init__()
        # gaussian
        self.mean = opt['noise']['GaussianNoise']['mean']
        self.min_val =  opt['noise']['GaussianNoise']['min_variance']
        self.max_val =  opt['noise']['GaussianNoise']['max_variance']
        self.device = device

    def gaussian_noise(self, image, mean, var):
        noise = torch.Tensor(np.random.normal(mean, var, image.shape)/128.).to(self.device)
        out = image + noise
        return out

    def forward(self, encoded, cover_img=None):
        self.var = np.random.rand() * (self.max_val - self.min_val) + self.min_val
        return self.gaussian_noise(encoded, self.mean, self.var)

