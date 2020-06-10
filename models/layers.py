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

class OldBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(OldBatchNorm2d, self).__init__()
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
        self.running_var = self.running_var.to(x.device)
        self.running_mean= self.running_mean.to(x.device)

        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True).to(x.device)
            var = (x - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True).to(x.device)
            # print("mean device:",mean.device)
            # print("runing device:",self.running_mean.device)
            self.running_mean = (self.momentum) * self.running_mean + (1-self.momentum) * mean

            self.running_var = (self.momentum) * self.running_var + (1 - self.momentum) * var
            # var = torch.max(self.limit,var)
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
            # self.running_var = (self.momentum) * self.running_var + (1 - self.momentum) * var

        else:
            # var = torch.max(self.limit,self.running_var)
            x = self.gamma * (x - self.running_mean) / (torch.sqrt(self.running_var +
                                                                   self.eps)) + self.beta
        return x
class DetachVarKeepMaxMinGrad(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.01, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(DetachVarKeepMaxMinGrad, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.total = 1

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean(dim=(0, 2, 3), keepdim=True)

            channelMax = \
            torch.max(torch.max(torch.max(input, 0, keepdim=True)[0], 2, keepdim=True)[0], 3, keepdim=True)[0]
            channelMin = \
            torch.min(torch.min(torch.min(input, 0, keepdim=True)[0], 2, keepdim=True)[0], 3, keepdim=True)[0]

            A = (input >= channelMax).float()
            # print("A:")
            # print(A)
            B = (input <= channelMin).float()
            Mask = A + B
            n = torch.tensor(input.numel() / (input.size(1)))
            input_ = input * Mask + (input * (1 - Mask)).detach()

            var = (input_ - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True)
            # var = torch.clamp(var,min=0.05,max=4)
            mean = mean.squeeze()
            var = var.squeeze()

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean  \
                                   + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var

            input = (input- mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps)).detach()
        else:
            mean = self.running_mean
            var = self.running_var
            input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return input


class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,running_mean,running_var, eps, momentum):
        n = x.numel() / (x.size(1))

        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        # mean = torch.clamp(mean,min=0,max=4)
        # print('mean size:', mean.size())
        # use biased var in train

        var = (x - mean).pow(2).sum(dim=(0, 2, 3))/(n)
        mean = mean.squeeze()
        var = var.squeeze()

        running_mean.copy_(momentum * mean\
                            + (1 - momentum) * running_mean)
        # update running_var with unbiased var
        running_var.copy_(momentum * var * n / (n - 1) \
                           + (1 - momentum) * running_var)
        y = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None]))
        ctx.eps = 0.00
        ctx.save_for_backward(y, var, )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # print("grad dtype")
        
        eps = ctx.eps
        y, var= ctx.saved_variables
        n = y.numel()/y.size(1)
        # print('y dtype',y.dtype)
        g = grad_output
        # print("g:",g[:,0,:,:])
        gy = (g * y).sum(dim=(0,2,3),keepdim=True)/(n)*y
        # print("gy dtype",gy.dtype)
        # print("g*y",(g * y).mean(dim=(0,2,3),keepdim=True)[:,0,:,:])
        # print("gy:",gy[:,0,:,:])
        g1 = g.mean(dim=(0,2,3),keepdim=True)
        # print("g1:",g1[:,0,:,:])
        gx_ = g - g1 - gy
        # print("g - g1",(g-g1)[:,0,:,:])
        # print("gx_:",gx_[:,0,:,:])
        gx = 1. / torch.sqrt(var[None, :, None, None]) * (gx_)
        # gx = gx.float()
        # print("gx:",gx[:,0,:,:])
        return gx, None, None,None,None

class GradBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(GradBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
    def forward(self,x):
        self._check_input_dim(x)

        exponential_average_factor = 0.1
        x = x.double()
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            y = BatchNormFunction.apply(x,self.running_mean,self.running_var,self.eps,self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
            y = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        y = y.float()
        return y