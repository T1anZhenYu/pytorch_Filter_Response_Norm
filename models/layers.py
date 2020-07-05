import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
import os
import math
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
            bias, weight, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(NewFilterResponseNormalization, self).__init__()
        self.bias = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.weight = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.limit = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
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

        assert (self.weight.shape[1],
                self.bias.shape[1], self.tau.shape[1]) == (c, c, c)
        if setting.temp_epoch / setting.total_epoch <= start:
            x = torch.max(self.weight * x + self.bias, self.tau)
        else :
            a = x.pow(2).mean(dim=(2, 3), keepdim=True)
            alpha =(setting.temp_epoch / setting.total_epoch)/(end-start) \
                   - start/(end-start)

            A = torch.max(self.limit, alpha * a +
                          torch.abs(self.eps))

            x = x / torch.sqrt(A + 1e-6)
            x = torch.max(self.weight * x + self.bias, self.tau)
        return x



class OldFilterResponseNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        Input Variables:
        ----------------
            bias, weight, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(OldFilterResponseNormalization, self).__init__()
        self.bias = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.weight = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.tau)
        nn.init.constant_(self.eps, 1e-4)
    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """

        n, c, h, w = x.shape
        assert (self.weight.shape[1],
                self.bias.shape[1], self.tau.shape[1]) == (c, c, c)
        A = x.pow(2).mean(dim=(2, 3), keepdim=True)
        x = x / torch.sqrt(A + 1e-6 + torch.abs(self.eps))
        x = torch.max(self.weight * x + self.bias, self.tau)
        return x

# class OldBatchNorm2d(nn.Module):
#     def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
#         """
#         Input Variables:
#         ----------------
#             bias, weight, tau: Variables of shape [1, C, 1, 1].
#             eps: A scalar constant or learnable variable.
#         """
#
#         super(OldBatchNorm2d, self).__init__()
#         self.affine = affine
#         if self.affine:
#             self.bias = nn.parameter.Parameter(
#                 torch.Tensor(1, num_features, 1, 1), requires_grad=True)
#             self.weight = nn.parameter.Parameter(
#                 torch.Tensor(1, num_features, 1, 1), requires_grad=True)
#         else:
#             self.bias = nn.parameter.Parameter(
#                 torch.Tensor(1, num_features, 1, 1), requires_grad=False)
#             self.weight = nn.parameter.Parameter(
#                 torch.Tensor(1, num_features, 1, 1), requires_grad=False)
#         self.eps = eps
#         self.running_mean = torch.zeros(1, num_features, 1, 1)
#         self.running_var = torch.ones(1, num_features, 1, 1)
#         # self.running_var = torch.Tensor(1, num_features, 1, 1)
#         self.limit = nn.parameter.Parameter(
#                 torch.Tensor(1, num_features, 1, 1), requires_grad=True)
#
#         self.momentum = momentum
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.ones_(self.weight)
#         nn.init.zeros_(self.bias)
#         nn.init.ones_(self.running_var)
#         nn.init.constant_(self.limit,0.1)
#
#     def forward(self, x):
#         self.running_var = self.running_var.to(x.device).double()
#         self.running_mean= self.running_mean.to(x.device).double()
#         x = x.double()
#         n = torch.tensor(x.numel() / (x.size(1)))
#         if self.training:
#             mean = x.mean(dim=(0, 2, 3), keepdim=True)
#             var = (x - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True)
#             # print("mean device:",mean.device)
#             # print("running mean device:",self.running_mean.device)
#             # print("runing device:",self.running_mean.device)
#             with torch.no_grad():
#                 self.running_mean = self.momentum * mean  \
#                                    + (1 - self.momentum) * self.running_mean
#                 self.running_mean = self.running_mean.float()
#                 # update running_var with unbiased var
#                 self.running_var = self.momentum * var * n / (n - 1) \
#                                    + (1 - self.momentum) * self.running_var
#                 self.running_var = self.running_var.float()
#             # var = torch.max(self.limit,var)
#             x = (x - mean) / torch.sqrt(var + self.eps)
#             # self.running_var = (self.momentum) * self.running_var + (1 - self.momentum) * var
#
#         else:
#             # var = torch.max(self.limit,self.running_var)
#             x = (x - self.running_mean) / (torch.sqrt(self.running_var +
#                                                                    self.eps))
#         if self.affine:
#             x = x * self.weight.double() + self.bias.double()
#         x = x.float()
#         return x
#
#
# class DetachVarKeepMaxMinFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, running_mean, running_var, eps, momentum):
#         n = x.numel() / (x.size(1))
#
#         mean = x.mean(dim=(0, 2, 3), keepdim=True)
#         # mean = torch.clamp(mean,min=0,max=4)
#         # print('mean size:', mean.size())
#         # use biased var in train
#
#         var = (x - mean).pow(2).sum(dim=(0, 2, 3)) / (n)
#         var = torch.clamp(var, min=0.05, max=4)
#         mean = mean.squeeze()
#         var = var.squeeze()
#
#         running_mean.copy_(momentum * mean \
#                            + (1 - momentum) * running_mean)
#         # update running_var with unbiased var
#         running_var.copy_(momentum * var * n / (n - 1) \
#                           + (1 - momentum) * running_var)
#         y = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None]))
#         ctx.eps = 0.00
#         ctx.save_for_backward(y, var,x )
#         return y
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # print("grad dtype")
#
#         eps = ctx.eps
#         y, var,x = ctx.saved_variables
#         n = y.numel() / y.size(1)
#         channelMax = \
#             torch.max(torch.max(torch.max(x, 0, keepdim=True)[0], 2, keepdim=True)[0], 3, keepdim=True)[0]
#         channelMin = \
#             torch.min(torch.min(torch.min(x, 0, keepdim=True)[0], 2, keepdim=True)[0], 3, keepdim=True)[0]
#
#         A = (y >= channelMax).double()
#         # print("A:")
#         # print(A)
#         B = (y <= channelMin).double()
#         Mask = A + B
#         # print('mask',Mask.shape)
#         g = grad_output
#         # print("g:",g[:,0,:,:])
#         gy = (g * y).sum(dim=(0, 2, 3), keepdim=True) / (n) * y
#         # print("gy dtype",gy.dtype)
#         # print("g*y",(g * y).mean(dim=(0,2,3),keepdim=True)[:,0,:,:])
#         # print("gy:",gy[:,0,:,:])
#         g1 = g.mean(dim=(0, 2, 3), keepdim=True)
#         # print("g1:",g1[:,0,:,:])
#         gx_ = g - g1 - gy * Mask
#         # print("g - g1",(g-g1)[:,0,:,:])
#         # print("gx_:",gx_[:,0,:,:])
#         gx = 1. / torch.sqrt(var[None, :, None, None]) * (gx_)
#         # gx = gx.float()
#         # print("gx:",gx[:,0,:,:])
#         return gx, None, None, None, None
#
#
# class DetachVarKeepMaxMin(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=0.00001, momentum=0.1,
#                  affine=False, track_running_stats=True):
#         super(DetachVarKeepMaxMin, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)
#
#     def forward(self, x):
#         self._check_input_dim(x)
#         self.running_mean = self.running_mean.double()
#         self.running_var = self.running_var.double()
#         exponential_average_factor = 0.1
#         x = x.double()
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#         if self.training:
#             y = DetachVarKeepMaxMinFunction.apply(x, self.running_mean, self.running_var, self.eps, self.momentum)
#         else:
#             mean = self.running_mean
#             var = self.running_var
#             y = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
#         if self.affine:
#             y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
#         self.running_var = self.running_var.float()
#         self.running_mean = self.running_mean.float()
#         y = y.float()
#         return y


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




class RangeBN(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        """
        Input Variables:
        ----------------
            bias, weight, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(RangeBN, self).__init__()
        self.affine = affine
        if self.affine:
            self.bias = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)
            self.weight = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        else:
            self.bias = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=False)
            self.weight = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=False)
        self.eps = eps
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        # self.running_var = torch.Tensor(1, num_features, 1, 1)
        self.uplimit = nn.parameter.Parameter(
                torch.DoubleTensor(num_features), requires_grad=True)
        self.downlimit = nn.parameter.Parameter(
                torch.DoubleTensor( num_features), requires_grad=True)

        self.momentum = momentum
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.running_var)
        nn.init.zeros_(self.running_mean)
        nn.init.constant_(self.downlimit,0.1)
        nn.init.constant_(self.uplimit, 5)
    def forward(self, x):
        # self._check_input_dim(x)
        self.running_mean = self.running_mean.double().to(x.device)
        self.running_var = self.running_var.double().to(x.device)
        x = x.double()
        n = x.numel() / (x.size(1))
        if self.training:
            # print('x', x)

            channelMax = \
                torch.max(torch.max(torch.max(x, 0)[0], -1, )[0], -1, )[0]
            channelMin = \
                torch.min(torch.min(torch.min(x, 0)[0], -1, )[0], -1, )[0]
            # print(channelMax.shape)
            var = (torch.pow((channelMax - channelMin), 2)) / (2 * math.log(n))

            mean = x.mean(dim=(0, 2, 3))
            # print(var.shape)
            self.running_mean.copy_(self.momentum * mean \
                                    + (1 - self.momentum) * self.running_mean)
            # update running_var with unbiased var
            self.running_var.copy_(self.momentum * var \
                                   + (1 - self.momentum) * self.running_var)
            y = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        else:
            mean = self.running_mean
            var = self.running_var
            y = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        y = y.float()
        return y


class OfficialDetachVar(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        """
        Input Variables:
        ----------------
            bias, weight, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(OfficialDetachVar, self).__init__()
        self.affine = affine
        if self.affine:
            self.bias = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)
            self.weight = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        else:
            self.bias = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=False)
            self.weight = nn.parameter.Parameter(
                torch.Tensor(1, num_features, 1, 1), requires_grad=False)
        self.eps = eps
        self.running_mean = torch.zeros(1, num_features, 1, 1)
        self.running_var = torch.ones(1, num_features, 1, 1)
        # self.running_var = torch.Tensor(1, num_features, 1, 1)
        self.uplimit = nn.parameter.Parameter(
                torch.DoubleTensor(1, num_features, 1, 1), requires_grad=True)
        self.downlimit = nn.parameter.Parameter(
                torch.DoubleTensor(1, num_features, 1, 1), requires_grad=True)

        self.momentum = momentum
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.running_var)
        nn.init.zeros_(self.running_mean)
        nn.init.constant_(self.downlimit,0.1)
        nn.init.constant_(self.uplimit, 5)
    def forward(self, x):
        self.running_var = self.running_var.to(x.device).double()
        self.running_mean= self.running_mean.to(x.device).double()
        x = x.double()
        n = torch.tensor(x.numel() / (x.size(1)))
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True)).detach()
            var = torch.min(var,self.downlimit)
            var = torch.max(var,self.uplimit)
            # print("mean device:",mean.device)
            # print("running mean device:",self.running_mean.device)
            # print("runing device:",self.running_mean.device)
            with torch.no_grad():
                self.running_mean = self.momentum * mean  \
                                   + (1 - self.momentum) * self.running_mean
                self.running_mean = self.running_mean.float()
                # update running_var with unbiased var
                self.running_var = self.momentum * var * n / (n - 1) \
                                   + (1 - self.momentum) * self.running_var
                self.running_var = self.running_var.float()
            # var = torch.max(self.limit,var)
            x = (x - mean) / torch.sqrt(var + self.eps)
            # self.running_var = (self.momentum) * self.running_var + (1 - self.momentum) * var

        else:
            # var = torch.max(self.limit,self.running_var)
            x = (x - self.running_mean) / (torch.sqrt(self.running_var +
                                                                   self.eps))
        if self.affine:
            x = x * self.weight.double() + self.bias.double()
        x = x.float()
        return x


class VarLearn(nn.Module):
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

        super(VarLearn, self).__init__()

        assert affine==False, 'NOT Support Affine Yet'
        # constant
        self.eps = eps
        self.initvalue = initvalue
        self.momentum = momentum
        # print("var initvalue:",self.initvalue)
        # buffer
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer("runnig_index",torch.ones(num_features))
        # parameter
        self.register_parameter('trainable_var',
            nn.Parameter(self.initvalue*torch.ones(num_features)))
        self.register_parameter('real_var',
            nn.Parameter(torch.ones(num_features)))

    def forward(self, x):
        n = x.numel() / (x.size(1))
        if self.training:
            mean = x.mean(dim=(0, 2, 3))

            # update the running mean and var
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1-self.momentum).add_((self.momentum) * self.trainable_var)

            self.trainable_var.data = torch.max(self.trainable_var.data,torch.Tensor([0.1]).to(x.device))
            var = self.trainable_var
            self.real_var.data = 0.1*((x - mean[None, :, None, None]).pow(2).mean(dim=(0, 2, 3)))+\
            self.real_var.data*0.9

            # y = (x - mean[None, :, None, None]) \
            #     / (torch.sqrt(torch.sqrt(self.trainable_var[None, :, None, None])) + self.eps)
            x.\
            sub_(mean[None, :, None, None]).\
            div_(torch.pow(var[None, :, None, None], exponent=1/4.) + self.eps)

        else:
            mean = self.running_mean
            var = self.running_var
            x.\
            sub_(mean[None, :, None, None]).\
            div_(torch.pow(var[None, :, None, None], exponent=1/4.) + self.eps)

        return x

class MixVar(nn.Module):
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

        super(MixVar, self).__init__()

        assert affine==False, 'NOT Support Affine Yet'
        # constant
        self.eps = eps

        self.momentum = momentum
        # print("var initvalue:",self.initvalue)
        # buffer
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('running_index', torch.ones(num_features))
        self.mixlayer = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        # nn.init.constant_(self.mixlayer.weight,1/num_features)
        # parameter


    def forward(self, x):
        n = x.numel() / (x.size(1))
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = (x - mean[None, :, None, None]).pow(2).mean(dim=(0, 2, 3))
            index = self.sigmoid(self.mixlayer(var[None,None,:])).squeeze()
            # update the running mean and var
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
            self.running_index.mul(1 - self.momentum).add_(self.momentum * index)

            # y = (x - mean[None, :, None, None]) \
            #     / (torch.sqrt(torch.sqrt(self.trainable_var[None, :, None, None])) + self.eps)
            x.\
            sub_(mean[None, :, None, None]).\
            div_(torch.pow(var[None, :, None, None], exponent=1/2.) + self.eps)
            x.mul_(index[None, :, None, None])
        else:
            mean = self.running_mean
            var = self.running_var
            index = self.sigmoid(self.mixlayer(var[None,None,:])).squeeze()
            x.\
            sub_(mean[None, :, None, None]).\
            div_(torch.pow(var[None, :, None, None], exponent=1/2.) + self.eps)
            x.mul_(index[None, :, None, None])
        return x
