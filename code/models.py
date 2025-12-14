import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ConvBlock3D(nn.Module):
    def __init__(self, in_channel=16, out_channel=32):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel,
                      kernel_size=(3,3,3),
                      stride=(1,1,1),
                      padding='same'),
            nn.BatchNorm3d(out_channel),
            nn.ReLU())
    
    def forward(self, x):
        y = self.block(x)
        return y

class UpConvBlock3D(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.up_conv_block = nn.Sequential(
            nn.ConvTranspose3d(input_channel,
                      output_channel,
                      kernel_size =(1,3,3),
                      stride=(1,2,2),
                      padding=(0,1,1),
                      output_padding=(0,1,1)),
            nn.Conv3d(output_channel,
                      output_channel,
                      kernel_size=(1,3,3),
                      padding=(0,1,1)),
            nn.BatchNorm3d(output_channel),
            nn.ReLU())

    def forward(self, x):
        y = self.up_conv_block(x)
        return y


class CNN3D(nn.Module):
    """
    Implementation of wide-deep network similar to proposed version on paper 
    Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
    """
    def __init__(self, in_channel=16, out_channel=32, temp_pool=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.temp_pool = temp_pool

        self.block1 = nn.Sequential(ConvBlock3D(in_channel=3, out_channel=self.in_channel),
                                    ConvBlock3D(in_channel=self.in_channel, out_channel=self.in_channel),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))

        self.block2 = nn.Sequential(ConvBlock3D(in_channel=self.in_channel, out_channel=self.out_channel),
                                    ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))

        self.block3 = nn.Sequential(ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))

        self.block4 = nn.Sequential(ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel),
                                    nn.MaxPool3d(kernel_size=(self.temp_pool,2,2), stride=(self.temp_pool,2,2)))

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)

        return y
    
class CameraClassifier(nn.Module):
    def __init__(self, n_classes=13, temp_pool=5):
        super().__init__()
        self.n_classes = n_classes
        self.temp_pool = temp_pool
        self.backbone = CNN3D(temp_pool=temp_pool)

        self.classifier = nn.Sequential(nn.Conv3d(self.backbone.out_channel, 16, kernel_size=(1,3,3)),
                    nn.ReLU(),
                    nn.Conv3d(16,16,kernel_size=(1,3,3)),
                    nn.ReLU(),
                    nn.Conv3d(16, self.n_classes, kernel_size=(1,1,1)),
                    nn.AdaptiveAvgPool3d(output_size=(1,1,1)))
        
    def forward(self, x):
        feat = self.backbone(x)
        logit = self.classifier(feat)

        return logit
    
class CNN3DRecon(nn.Module):
    def __init__(self, in_channel:int=32, out_channel:int=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.encoder = CNN3D(in_channel=in_channel, out_channel=out_channel, temp_pool=1)

        self.decoder = nn.Sequential(
            UpConvBlock3D(out_channel, out_channel//4),
            UpConvBlock3D(out_channel//4, out_channel//8),
            UpConvBlock3D(out_channel//8, 3),
            nn.ConvTranspose3d(3,3,kernel_size =(1,3,3),
                                  stride=(1,2,2),
                                  padding=(0,1,1),
                                  dilation=(1,5,4),
                                  output_padding=(0,1,1)))
        
    def forward(self, x):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)

        return recon




