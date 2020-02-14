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
        self.limit = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
        nn.init.constant_(self.limit,0.5)
        nn.init.constant_(self.eps,1e-4)
    def forward(self, x,start=0,end=1):
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

            A = torch.max(self.limit, alpha * a +
                          torch.abs(self.eps))

            x = x / torch.sqrt(A + 1e-6)
            x = torch.max(self.gamma * x + self.beta, self.tau)
        return x
class noalpha(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(noalpha, self).__init__()
        self.beta = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.limit = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.total = 0
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
        nn.init.constant_(self.limit,0.5)
        nn.init.constant_(self.eps,1e-4)
    def forward(self, x,start=0,end=1):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        n, c, h, w = x.shape
        assert (self.gamma.shape[1],
                self.beta.shape[1], self.tau.shape[1]) == (c, c, c)

        A = x.pow(2).mean(dim=(2, 3), keepdim=True)
        # self.total = self.total + 1
        # if h == 32 and self.total %300 == 1 and self.training:
        #     print("saving")
        #
        #     dic = {}
        #     dic['eps']=self.limit.cpu().detach().numpy()
        #     dic['var']=A.cpu().detach().numpy()
        #     dic['alpha*var']=(A).cpu().detach().numpy()
        #     np.savez("./npz/"+str(self.total)+"tempiter",**dic)
        A = torch.max(self.limit, A + torch.abs(self.eps))

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

class NewBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(NewBatchNorm2d, self).__init__()
        self.affine = affine
        if self.affine:
            self.beta = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)
            self.gamma = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        else:
            self.beta = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=False)
            self.gamma = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=False)
        self.eps = eps
        self.running_mean = torch.zeros(1, num_features, 1, 1)
        self.running_var = torch.ones(1, num_features, 1, 1)
        # self.running_var = torch.Tensor(1, num_features, 1, 1)
        self.limit = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)

        self.momentum = momentum
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.running_var)
        nn.init.constant_(self.limit,0.1)

    def forward(self, x):

        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True).to(x.device)
            var = (x - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True).to(x.device)
            self.running_var = (self.momentum) * self.running_var + (1 - self.momentum) * var
            # var = torch.max(self.limit,var)
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
            # self.running_var = (self.momentum) * self.running_var + (1 - self.momentum) * var

            self.running_mean = (self.momentum) * self.running_mean + (1-self.momentum) * mean
        else:

            x = self.gamma * (x - self.running_mean) / (torch.sqrt(self.running_var +
                                                                   self.eps)) + self.beta
        return x