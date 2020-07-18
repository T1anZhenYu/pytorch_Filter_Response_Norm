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

    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        super(MixChannel, self).__init__()
        ks = 5
        
        # self.mixvar = nn.Conv2d(1, 1, kernel_size=ks , padding=(ks-1) // 2, bias=False) 
        self.linearvar = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        # self.mixmean = nn.Conv2d(1, 1, kernel_size=ks , padding=(ks-1) // 2, bias=False) 
        self.linearmean = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        # self.mixmean = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        self.combine = nn.Conv2d(2,1,kernel_size=ks, padding=(ks-1) // 2,, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(num_features)


    def forward(self, x):
        n = x.numel() / (x.size(1))
        if self.training:
            
            mean = x.mean(dim=(0, 2, 3))
            var = (x-mean[None, :, None, None]).pow(2).mean(dim=(0,2, 3))

            indexvar = torch.sqrt(self.bn.running_var).mean()/math.pow(n,0.5)
            indexmean = self.bn.running_mean.mean()/math.pow(n,0.5)

            varmix = indexvar * torch.sqrt(var)
            varmix = self.sigmoid(self.linearvar(varmix[None,None,:]).squeeze())

            meanmix = indexmean * mean 

            meanmix = self.sigmoid(self.linearmean(meanmix[None,None,:]).squeeze())
            
            combine = torch.cat((varmix[None,None,None,:],meanmix[None,None,None,:]),1)

            index = self.combine(combine).squeeze()
            # print("index:",index.shape)
            # print(meanmix.shape)
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var

            indexvar = torch.sqrt(self.bn.running_var).mean()/math.pow(n,0.5)
            indexmean = self.bn.running_mean.mean()/math.pow(n,0.5)

            varmix = indexvar * torch.sqrt(var)
            varmix = self.sigmoid(self.linearvar(varmix[None,None,:]).squeeze())

            meanmix = indexmean * mean 

            meanmix = self.sigmoid(self.linearmean(meanmix[None,None,:]).squeeze())
            
            combine = torch.cat((varmix[None,None,None,:],meanmix[None,None,None,:]),1)

            index = self.combine(combine).squeeze()
        out = self.bn(x)
        out.mul_(index[None,:,None,None])
        return out

