import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .. import layers as L


__all__ = ['ResNetFRN', 'l_resnetfrn18', 'l_resnetfrn34', 'l_resnetfrn50', 'l_resnetfrn101',
           'l_resnetfrn152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return L.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return L.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = L.FilterResponseNormalization(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = L.FilterResponseNormalization(planes)
        self.bn3 = L.FilterResponseNormalization(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        epoch = epoch_g
        total_epoch = total_epoch_g

        identity = x

        out = self.conv1(x)
        out = self.bn1(out, epoch, total_epoch)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, epoch, total_epoch)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn3(identity, epoch, total_epoch)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = L.FilterResponseNormalization(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = L.FilterResponseNormalization(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = L.FilterResponseNormalization(planes * self.expansion)
        self.bn4 = L.FilterResponseNormalization(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        epoch = epoch_g
        total_epoch = total_epoch_g

        identity = x

        out = self.conv1(x)
        out = self.bn1(out,epoch,total_epoch)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,epoch,total_epoch)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out,epoch,total_epoch)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn4(identity,epoch,total_epoch)

        out += identity
        out = self.relu(out)

        return out


class ResNetFRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetFRN, self).__init__()
        self.inplanes = 64
        self.conv1 = L.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = L.FilterResponseNormalization(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # L.FilterResponseNormalization(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, epoch, total_epoch):
        global epoch_g
        epoch_g = epoch
        global  total_epoch_g
        total_epoch_g = total_epoch

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def l_resnetfrn18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFRN(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def l_resnetfrn34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFRN(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def l_resnetfrn50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFRN(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def l_resnetfrn101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFRN(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def l_resnetfrn152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFRN(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
