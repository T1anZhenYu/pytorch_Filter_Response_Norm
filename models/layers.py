import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
import os
import math
import setting

class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,running_mean,running_var, eps, momentum):
        n = x.numel() / (x.size(1))
        #running_mean = running_mean.double()
        #running_var = running_var.double()        
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
        #running_mean = running_mean.float()
        #running_var = running_var.float()
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
                 affine=True, track_running_stats=True):
        super(GradBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
    def forward(self,x):
        self._check_input_dim(x)
        self.running_mean = self.running_mean.double()
        self.running_var = self.running_var.double()
        x = x.double()
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


class MixChannel(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, initvalue=25):
        """
        Input Variables:
        ----------------
            num_features: in channels
            eps: eps to avoid dviding zero
            momentum: momentum for updating of running var and mean
            affine: has to be False
            initvaule: initial value of trainable var
        """

        super(MixChannel, self).__init__()

        assert affine==False, 'NOT Support Affine Yet'
        # constant
        self.eps = eps

        self.momentum = momentum
        # print("var initvalue:",self.initvalue)
        # buffer
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        ks = 5
        
        self.mixvar = nn.Conv2d(1, 1, kernel_size=ks , padding=(ks-1) // 2, bias=False) 
        self.linearvar = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        self.mixmean = nn.Conv2d(1, 1, kernel_size=ks , padding=(ks-1) // 2, bias=False) 
        self.linearmean = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        # self.mixmean = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        # nn.init.constant_(self.mixlayer.weight,1/num_features)
        # parameter


    def forward(self, x):
        n = x.numel() / (x.size(1))
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = (x - mean[None, :, None, None]).pow(2).mean(dim=(0, 2, 3))

            # update the running mean and var
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)

            varmix = (torch.mm(torch.sqrt(self.running_var[:,None]).detach(),torch.sqrt(var[None,:])))/math.pow(x.size(1),0.5)
            # print(varmix.shape)
            # varmix = self.mixvar(varmix[None,None,:,:])
            varmix = varmix.mean(dim=(0))
            # print(varmix.shape)
            varmix = self.sigmoid(self.linearvar(varmix[None,None,:]).squeeze())
            # print(varmix.shape)
            meanmix = (torch.mm(self.running_mean[:,None].detach(),mean[None,:]))/math.pow(x.size(1),0.5)
            # print(meanmix.shape)
            # meanmix = self.mixmean(meanmix[None,None,:,:])
            meanmix = meanmix.mean(dim=(0))
            # print(meanmix.shape)
            meanmix = self.sigmoid(self.linearmean(meanmix[None,None,:]).squeeze())
            # print(meanmix.shape)

            x.\
            sub_(mean[None, :, None, None]).\
            div_(torch.pow(var[None, :, None, None], exponent=1/2.) + self.eps)
            x.mul_(0.5*varmix[None, :, None, None]+0.5*meanmix[None, :, None, None])
        else:
            mean = self.running_mean
            var = self.running_var
            varmix = (torch.mm(torch.sqrt(self.running_var[:,None]).detach(),torch.sqrt(var[None,:])))\
            /math.pow(x.size(1),0.5)
            # print(varmix.shape)
            # varmix = self.mixvar(varmix[None,None,:,:])
            varmix = varmix.mean(dim=(0))
            # print(varmix.shape)
            varmix = self.sigmoid(self.linearvar(varmix[None,None,:]).squeeze())
            # print(varmix.shape)
            meanmix = (torch.mm(self.running_mean[:,None].detach(),mean[None,:]))/math.pow(x.size(1),0.5)
            # print(meanmix.shape)
            # meanmix = self.mixmean(meanmix[None,None,:,:])
            meanmix = meanmix.mean(dim=(0))
            # print(meanmix.shape)
            meanmix = self.sigmoid(self.linearmean(meanmix[None,None,:]).squeeze())
            x.\
            sub_(mean[None, :, None, None]).\
            div_(torch.pow(var[None, :, None, None], exponent=1/2.) + self.eps)
            x.mul_(0.5*varmix[None, :, None, None]+0.5*meanmix[None, :, None, None])
       
        return x
