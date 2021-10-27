"""
TODO:
- Try out groupnorm
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    卷积操作
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, name=None):
        super(ConvBlock, self).__init__()

        block = []
        # first conv layer
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, stride=stride))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_channels))
        # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，
        # 这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定

        # second conv layer
        block.append(nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding=padding, stride=stride))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_channels))

        # make sequential
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        output = self.conv_block(x)

        return output


class DownSampling(nn.Module):
    """
    下采样
    """

    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super(DownSampling, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        output = self.max_pool(conv_out)

        return output, conv_out


class UpSampling(nn.Module):
    """
    上采样
    """

    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super(UpSampling, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel_size)
        self.conv_t = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=1, stride=2, output_padding=1)

    def forward(self, x, skip):
        conv_out = self.conv(x)
        output = self.conv_t(conv_out)

        output += skip

        return output

