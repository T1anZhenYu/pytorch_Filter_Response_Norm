from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
from ..layers import FilterResponseNormalization

__all__ = ['resnet_frn']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.frn1 = FilterResponseNormalization(planes)
        # self.tlu1 = TLU(planes)

        self.frn2 = FilterResponseNormalization(planes)
        # self.tlu2 = TLU(planes)


    def forward(self, x):
        lr =  lr_g
        lr_max = lr_max_g
        residual = x

        out = self.conv1(x)
        out = self.frn1(out, lr, lr_max)
        out = self.conv2(out)
        out = self.frn2(out, lr, lr_max)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Frn(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet_Frn, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        # if block_name.lower() == 'basicblock':
        #     assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
        #     n = (depth - 2) // 6
        #     block = BasicBlock
        # elif block_name.lower() == 'bottleneck':
        #     assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
        #     n = (depth - 2) // 9
        #     block = Bottleneck
        # else:
        #     raise ValueError('block_name shoule be Basicblock or Bottleneck')
        assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
        n = (depth - 2) // 6
        block = BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.frn1 = FilterResponseNormalization(16)
        # self.tlu1 = TLU(16)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, lr, lr_max):
        global lr_g
        lr_g = lr
        global  lr_max_g
        lr_max_g = lr_max

        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)    # 32x32
        x = self.frn1(x, lr, lr_max)
        # x = self.tlu1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x) # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return x


def resnet_frn(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet_Frn(**kwargs)
