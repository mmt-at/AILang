import torch
import torch.nn.functional as F
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

class TorchResNet(torch.nn.Module):
    def __init__(self):
        super(TorchResNet, self).__init__()
        self.in_planes = 64

        # 初始的卷积和池化层
        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Layer 1
        self.conv2 = torch.nn.Conv2d(
            64, 64, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(
            64, 64, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()

        # Layer 2
        self.conv4 = torch.nn.Conv2d(
            64, 128, stride=2, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(
            128, 128, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn5 = torch.nn.BatchNorm2d(128)

        self.ds2_conv = torch.nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.ds2_bn = torch.nn.BatchNorm2d(128)
        self.relu3 = torch.nn.ReLU()

        # Layer 3
        self.conv6 = torch.nn.Conv2d(
            128, 256, stride=2, kernel_size=3, padding=1, bias=False
        )
        self.bn6 = torch.nn.BatchNorm2d(256)
        self.conv7 = torch.nn.Conv2d(
            256, 256, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn7 = torch.nn.BatchNorm2d(256)

        self.ds3_conv = torch.nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.ds3_bn = torch.nn.BatchNorm2d(256)
        self.relu4 = torch.nn.ReLU()

        # Layer 4
        self.conv8 = torch.nn.Conv2d(
            256, 512, stride=2, kernel_size=3, padding=1, bias=False
        )
        self.bn8 = torch.nn.BatchNorm2d(512)
        self.conv9 = torch.nn.Conv2d(
            512, 512, stride=1, kernel_size=3, padding=1, bias=False
        )
        self.bn9 = torch.nn.BatchNorm2d(512)

        self.ds4_conv = torch.nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.ds4_bn = torch.nn.BatchNorm2d(512)
        self.relu5 = torch.nn.ReLU()

    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        # return x
        # Layer 1
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity  # 残差连接
        out = self.relu2(out)
        # return out
        # Layer 2
        x = out
        identity2 = x
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu3(out)

        out = self.conv5(out)
        out = self.bn5(out)
        # return out

        # 下采样并加上残差连接
        # return identity2
        identity2 = self.ds2_conv(identity2)
        identity2 = self.ds2_bn(identity2)
        # return identity2
        out += identity2
        # return out
        out = self.relu3(out)
        # Layer 3
        x = out
        identity3 = x
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu4(out)

        out = self.conv7(out)
        out = self.bn7(out)

        identity3 = self.ds3_conv(identity3)
        identity3 = self.ds3_bn(identity3)
        out += identity3
        out = self.relu4(out)
        # Layer 4
        x = out
        identity4 = x
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu5(out)

        out = self.conv9(out)
        out = self.bn9(out)

        identity4 = self.ds4_conv(identity4)
        identity4 = self.ds4_bn(identity4)
        out += identity4
        out = self.relu4(out)
        return out

    
import ailang as al
from ailang import nn
import ailang.nn as nn

class AilangResNet(al.nn.Module):
    def __init__(self, pd):
        super().__init__()
        self.in_planes = 64

        # 初始的卷积和池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = al.from_numpy(pd["conv1.weight"])
        self.bn1 = nn.Batchnorm2d(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # layer 1
        self.conv2 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv2.weight = al.from_numpy(pd["conv2.weight"])
        self.bn2 = nn.Batchnorm2d(64)
        # 第二个卷积层
        self.conv3 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv3.weight = al.from_numpy(pd["conv3.weight"])

        self.bn3 = nn.Batchnorm2d(64)
        self.relu2 = nn.ReLU()
        # layer2
        self.conv4 = nn.Conv2d(64, 128, stride=2, kernel_size=3, padding=1, bias=False)
        self.conv4.weight = al.from_numpy(pd["conv4.weight"])
        self.bn4 = nn.Batchnorm2d(128)
        # 第二个卷积层
        self.conv5 = nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv5.weight = al.from_numpy(pd["conv5.weight"])

        self.bn5 = nn.Batchnorm2d(128)
        self.ds2_conv = nn.Conv2d(
                64,
                128 * 1,
                kernel_size=1,
                stride=2,
                bias=False,
            )
        self.ds2_conv.weight = al.from_numpy(pd["ds2_conv.weight"])
        self.ds2_bn = nn.Batchnorm2d(128 * 1)
        self.relu3 = nn.ReLU()
        # Layer 3
        self.conv6 = nn.Conv2d(128, 256, stride=2, kernel_size=3, padding=1, bias=False)
        self.conv6.weight = al.from_numpy(pd["conv6.weight"])
        self.bn6 = nn.Batchnorm2d(256)
        # 第二个卷积层
        self.conv7 = nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv7.weight = al.from_numpy(pd["conv7.weight"])

        self.bn7 = nn.Batchnorm2d(256)
        self.ds3_conv = nn.Conv2d(
                128,
                256 * 1,
                kernel_size=1,
                stride=2,
                bias=False,
            )
        self.ds3_conv.weight = al.from_numpy(pd["ds3_conv.weight"])
        self.ds3_bn = nn.Batchnorm2d(256 * 1)
        self.relu4 = nn.ReLU()

        # Layer 4
        self.conv8 = nn.Conv2d(256, 512, stride=2, kernel_size=3, padding=1, bias=False)
        self.conv8.weight = al.from_numpy(pd["conv8.weight"])
        self.bn8 = nn.Batchnorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv9.weight = al.from_numpy(pd["conv9.weight"])

        self.bn9 = nn.Batchnorm2d(512)
        self.ds4_conv = nn.Conv2d(
                256,
                512 * 1,
                kernel_size=1,
                stride=2,
                bias=False,
            )
        self.ds4_conv.weight = al.from_numpy(pd["ds4_conv.weight"])
        self.ds4_bn = nn.Batchnorm2d(512 * 1)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        # return x

        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = al.add(out, identity)
        out = self.relu2(out)
        # return out
        # layer2
        x = out
        identity2 = x
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu3(out)

        out = self.conv5(out)
        out = self.bn5(out)
        # return out

        # 如果存在下采样，调整输入的尺寸
        # return identity2
        identity2 = self.ds2_conv(identity2)
        identity2 = self.ds2_bn(identity2)
        # return identity2
        out = al.add(out, identity2)
        out = self.relu3(out)

        # Layer 3
        x = out
        identity3 = x
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu4(out)

        out = self.conv7(out)
        out = self.bn7(out)

        identity3 = self.ds3_conv(identity3)
        identity3 = self.ds3_bn(identity3)
        out = al.add(out, identity3)
        out = self.relu4(out)
        # Layer 4
        x = out
        identity4 = x
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu5(out)

        out = self.conv9(out)
        out = self.bn9(out)

        identity4 = self.ds4_conv(identity4)
        identity4 = self.ds4_bn(identity4)
        out = al.add(out, identity4)

        out = self.relu4(out)
        return out
