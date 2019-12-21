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


class MyFRN(torch.autograd.Function):
    @staticmethod
    def forward(self, x):

        A = x.pow(2).mean(dim=(2, 3), keepdim=True)
        x_hat = x / torch.sqrt(A + 1e-6)
        self.save_for_backward(x, x_hat, A)
        return x_hat

    @staticmethod
    def backward(self, grad_output):
        x, x_hat, A = self.saved_tensors

        B,C,W,H = x.shape
        part1 = torch.sum(torch.diagonal(torch.matmul(torch.transpose(x_hat,2,3),grad_output),
                                         dim1=2,dim2=3),dim=-1).view(B,C,1,1)

        d1 = torch.mul(x_hat,part1)

        d2 = torch.sum(grad_output,dim=(2,3),keepdim=True).repeat(1,1,W,H)

        d3 = (-d1 + d2)/(W*H)
        dx = d3 / torch.sqrt(A+1e-6)
        return dx, None, None

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
        self.frn = MyFRN.apply
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
    def forward(self, x, iter = 0):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """

        n, c, h, w = x.shape
        assert (self.gamma.shape[1],
                self.beta.shape[1], self.tau.shape[1]) == (c, c, c)

        x = self.frn(x)
        # A = x.pow(2).mean(dim=(2, 3), keepdim=True)
        # x_hat = x / torch.sqrt(A + 1e-6)
        x = torch.max(self.gamma*x + self.beta, self.tau)
        return x

