import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
import os
import math

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

class NewBN(nn.Module):
    def __init__(self,num_features, eps=1e-05, momentum=0.9, affine=True, gamma=2, b=1):
        super(NewBN,self).__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=affine)
        t = int(num_features/8)
        ks = t if t % 2 else t + 1
        
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(1,ks,kernel_size=(2,1),bias=False)
        self.conv1 =  nn.Conv1d(ks, 1, kernel_size=ks, padding=(ks-1) // 2, bias=False) 
        self.momentum = momentum
    def forward(self,x):
        n = x.numel() / (x.size(0))
        if self.training:
            # print("in train")
            mean = x.mean(dim=(0, 2, 3)).detach()
            var = (x-mean[None, :, None, None]).pow(2).mean(dim=(0,2, 3)).detach()

            indexvar = torch.sqrt(self.bn.running_var).mean()/math.pow(n,0.5)
            indexmean = self.bn.running_mean.mean()/math.pow(n,0.5)

            meanmix = indexmean * mean 
            varmix = indexvar * torch.sqrt(var)

            combine = torch.cat((meanmix[None,None,None,:],varmix[None,None,None,:]),2)

            combine = F.relu(self.conv0(combine)).squeeze()

            
            combine = self.sigmoid(self.conv1(combine[None,:,:])).squeeze()

        else:
            mean = self.bn.running_mean
            var = self.bn.running_var
            indexvar = torch.sqrt(self.bn.running_var).mean()/math.pow(n,0.5)
            indexmean = self.bn.running_mean.mean()/math.pow(n,0.5)

            meanmix = indexmean * mean 
            varmix = indexvar * torch.sqrt(var)

            combine = torch.cat((meanmix[None,None,None,:],varmix[None,None,None,:]),2)

            combine = F.relu(self.conv0(combine)).squeeze()

            
            combine = self.sigmoid(self.conv1(combine[None,:,:])).squeeze()

        out = self.bn(x)
        out.mul_(combine[None,:,None,None])
        return out   

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 
class EcaLayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

