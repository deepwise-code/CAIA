# -*- coding=utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DenseNet',]

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm3d(in_channel),
        nn.ReLU(),
        nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm3d(in_channel),
        nn.ReLU(),
        nn.Conv3d(in_channel, out_channel, 1),
        nn.AvgPool3d(2, 2)
    )
    return trans_layer

class DenseNet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(DenseNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=False),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        # self.block1 = nn.Sequential(
        #     nn.Conv3d(in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm3d(num_features=64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # )
        self.DB1 = self._make_dense_block(64, growth_rate, num=block_layers[0])
        self.TL1 = self._make_transition_layer(256)
        self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(512)
        self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(1024)
        # self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])

        self.global_average = nn.Sequential(
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Linear(512, num_classes)

        self.__init_params__()

    def forward(self, x):
        x = self.block1(x)
        x = self.DB1(x)
        x = self.TL1(x)
        x = self.DB2(x)
        x = self.TL2(x)
        x = self.DB3(x)
        x = self.TL3(x)
        # x = self.DB4(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)

    def _make_transition_layer(self, channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)

    def __init_params__(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
