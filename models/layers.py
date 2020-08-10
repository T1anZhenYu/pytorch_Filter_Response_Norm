import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
import os
import math
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

