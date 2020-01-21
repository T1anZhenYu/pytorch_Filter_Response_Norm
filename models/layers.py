import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
import os
import setting
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

class NewFilterResponseNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(NewFilterResponseNormalization, self).__init__()
        self.beta = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
        nn.init.constant_(self.eps,1e-4)
    def forward(self, x,start=0.1,end=0.9):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        n, c, h, w = x.shape
        assert (self.gamma.shape[1],
                self.beta.shape[1], self.tau.shape[1]) == (c, c, c)
        if setting.temp_epoch / setting.total_epoch <= start:
            x = torch.max(self.gamma * x + self.beta, self.tau)
        else :
            a = x.pow(2).mean(dim=(2, 3), keepdim=True)
            alpha =(setting.temp_epoch / setting.total_epoch)/(end-start) \
                   - start/(end-start)

            A = torch.max(torch.tensor(1.).to(x.device), alpha * a +
                          torch.abs(self.eps))

            x = x / torch.sqrt(A + 1e-6)
            x = torch.max(self.gamma * x + self.beta, self.tau)
        return x


class OldFilterResponseNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(OldFilterResponseNormalization, self).__init__()
        self.beta = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
        nn.init.constant_(self.eps, 1e-4)
    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """

        n, c, h, w = x.shape
        assert (self.gamma.shape[1],
                self.beta.shape[1], self.tau.shape[1]) == (c, c, c)
        A = x.pow(2).mean(dim=(2, 3), keepdim=True)
        x = x / torch.sqrt(A + 1e-6 + torch.abs(self.eps))
        x = torch.max(self.gamma * x + self.beta, self.tau)
        return x