import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
import os

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)


class MyMin(torch.autograd.Function):
    @staticmethod
    def forward(self, x, uplim, slope):
        self.save_for_backward(x, uplim,slope)

        output = (x <= uplim).float() * x + (x > uplim).float() * (slope * x + uplim * (1 - slope))
        return output

    @staticmethod
    def backward(self, grad_output):
        x, uplim, slope = self.saved_tensors

        dl_dx = (x <= uplim).float() * grad_output + (x > uplim).float() * slope * grad_output

        dl_duplime = (x > uplim).float() * (1 - slope) * grad_output



        return dl_dx, dl_duplime,None

class FilterResponseNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(FilterResponseNormalization, self).__init__()
        self.beta = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.uplim = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(torch.Tensor([eps]),requires_grad=False)
        self.min = MyMin.apply
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.ones_(self.uplim)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
    def forward(self, x, iter = 0):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """

        n, c, h, w = x.shape
        assert (self.gamma.shape[1], self.uplim.shape[1],
                self.beta.shape[1], self.tau.shape[1]) == (c, c, c, c)

        slope = 1 / torch.log(torch.tensor(h*w + 1e-6).to(x.device))

        x = x / slope
        x = torch.max(self.gamma*x + self.beta, self.tau)
        return x

class MaxMinFRN(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(MaxMinFRN, self).__init__()
        self.beta = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(torch.Tensor([eps]),requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """

        n, c, h, w = x.shape
        assert (self.gamma.shape[1], self.beta.shape[1], self.tau.shape[1]) == (c, c, c)

        # Compute the max of activations per channel
        channel_max = torch.max(torch.max(x,dim=2,keepdim=True)[0],dim=3,keepdim=True)[0]
        channel_min = torch.min(torch.min(x,dim=2,keepdim=True)[0],dim=3,keepdim=True)[0]

        Cn = torch.log(torch.tensor(h * w + 0.000001)) * 2
        sigma = (channel_max - channel_min).pow(2)/Cn
        mu = (channel_min + channel_max).pow(2)/4
        nu2 = sigma + mu
        # Perform FRN
        x = x * torch.rsqrt(nu2 + 1e-6 + torch.abs(self.eps))
        # Return after applying the Offset-ReLU non-linearity
        return torch.max(self.gamma*x + self.beta, self.tau)