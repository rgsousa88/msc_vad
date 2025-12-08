import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchvision.models import mobilenet_v3_large

class UpConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.up_conv_block = nn.Sequential(
            nn.ConvTranspose2d(input_channel,
                      output_channel,
                      kernel_size = 3,
                      stride = 2,
                      padding = 1,
                      output_padding = 1),
            nn.Conv2d(output_channel,
                      output_channel,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU())

    def forward(self, x):
        y = self.up_conv_block(x)
        return y
    
class MobileNetV3Encoder(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.model = mobilenet_v3_large(pretrained=pretrained).features
        self.depth = 5
        self.out_channels = [24, 40, 112, 160]
        self.stgBounds = [3, 5, 12, -1]
        self.setStages()
    
    def setStages(self,):
        self.stages = []
        start = 0
        for i in range(len(self.stgBounds)):
            self.stages.append(self.model[start:self.stgBounds[i]])
            start = self.stgBounds[i]
    
    def forward(self, x):
        features = []
        
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            features.append(x)
        
        return features

class MobV3ReconBase(nn.Module):
    def __init__(self, inputSize:int = 224, pretrained:bool = False):
        super().__init__()
        self.encoder = MobileNetV3Encoder(pretrained=pretrained)
        self.decoder = nn.Sequential(UpConvBlock(self.encoder.out_channels[-1],self.encoder.out_channels[-2]),
                                     UpConvBlock(self.encoder.out_channels[-2],self.encoder.out_channels[-3]),
                                     UpConvBlock(self.encoder.out_channels[-3],self.encoder.out_channels[-4]),
                                     UpConvBlock(self.encoder.out_channels[-4],3),
                                     nn.ConvTranspose2d(3, 3,
                                                        kernel_size = 3,
                                                        stride = 2,
                                                        padding = 1,
                                                        output_padding=1))
        
    def forward(self, x):
        features = self.encoder(x)
        y = self.decoder(features[-1])

        return y