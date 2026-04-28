import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

SEED = 122344
torch.manual_seed(SEED)
random.seed(SEED)

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
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
                                    ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel))
        

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = F.max_pool3d(y, kernel_size=(y.shape[2],2,2), stride=2)

        return y

class CNN3DRes(CNN3D):
    def __init__(self, in_channel=32, out_channel=64):
        super().__init__(in_channel=in_channel, out_channel=out_channel)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(y1)
        y3 = self.block3(y2) + self.pool1(y2)
        y = self.pool1(y3)

        return y

class CNN3DResSkip(CNN3D):
    def __init__(self, in_channel=32, out_channel=64, temp_pool=1):
        super().__init__(in_channel=in_channel, out_channel=out_channel, temp_pool=temp_pool)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.channel_adapter = nn.Conv3d(in_channel, out_channel, kernel_size=(1,1,1),stride=(1,1,1))

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(y1)
        y3 = self.block3(y2)
        
        y = self.channel_adapter(y1)
        y = F.interpolate(y, (y3.shape[2], y3.shape[3], y3.shape[4]))
        y = y + y3

        return y

class CNN3DResSkip2(CNN3D):
    def __init__(self, in_channel=32, out_channel=64, temp_pool=1):
        super().__init__(in_channel=in_channel, out_channel=out_channel, temp_pool=temp_pool)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.channel_adapter = nn.Conv3d(out_channel, in_channel, kernel_size=(1,1,1),stride=(1,1,1))

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(y1)
        y3 = self.block3(y2)
        
        y = self.channel_adapter(y3)
        y = F.interpolate(y, (y1.shape[2], y1.shape[3], y1.shape[4]))
        y = y + y1

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
                                  #dilation=(1,5,4),
                                  output_padding=(0,1,1)))
        
    def forward(self, x):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)

        return recon

class CNN3DResRecon(nn.Module):
    def __init__(self, in_channel:int=32, out_channel:int=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.encoder = CNN3DRes(in_channel=in_channel, out_channel=out_channel, temp_pool=1)

        self.decoder = nn.Sequential(
            UpConvBlock3D(out_channel, out_channel//4),
            UpConvBlock3D(out_channel//4, out_channel//8),
            UpConvBlock3D(out_channel//8, 3),
            nn.ConvTranspose3d(3,3,kernel_size =(1,3,3),
                                  stride=(1,2,2),
                                  padding=(0,1,1),
                                  #dilation=(1,5,4),
                                  output_padding=(0,1,1)))
        
    def forward(self, x):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)

        return recon

class CNN3DResReconSkipV1(nn.Module):
    def __init__(self, in_channel:int=32, out_channel:int=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.encoder = CNN3DResSkip(in_channel=in_channel, out_channel=out_channel, temp_pool=1)

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

class CNN3DResReconSkipV2(nn.Module):
    def __init__(self, in_channel:int=32, out_channel:int=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.encoder = CNN3DResSkip2(in_channel=in_channel, out_channel=out_channel, temp_pool=1)

        self.decoder = nn.Sequential(UpConvBlock3D(in_channel, in_channel//8),
                                     UpConvBlock3D(in_channel//8, 3),
                                     nn.Conv3d(3,3,kernel_size=(1,3,3),
                                                   stride=(1,2,2),
                                                   padding=(0,1,1)))
        
    def forward(self, x):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)

        return recon

class SSMTLRecon(nn.Module):
    def __init__(self, in_channel=64, out_channel=3):
        super().__init__()
        self.in_channel = in_channel
        self.inter_channel = in_channel // 2
        self.out_channel = out_channel

        self.block1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=self.inter_channel),
                                    nn.ReLU(),)

        self.block2 = nn.Sequential(nn.Conv2d(in_channels=self.inter_channel, out_channels=self.inter_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=self.inter_channel),
                                    nn.ReLU(),)

        self.block3 = nn.Sequential(nn.Conv2d(in_channels=self.inter_channel, out_channels=self.inter_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=self.inter_channel),
                                    nn.ReLU())

        self.block4 = nn.Sequential(nn.Conv2d(in_channels=self.inter_channel, out_channels=out_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=out_channel),
                                    nn.ReLU())
    
    def forward(self, x):
        y = F.interpolate(x, scale_factor=2)
        y = self.block1(y)

        y = F.interpolate(y, scale_factor=2)
        y = self.block2(y)

        y = F.interpolate(y, scale_factor=2)
        y = self.block3(y)
        
        y = F.interpolate(y, scale_factor=2)
        y = self.block4(y)

        return y
    
class SSMTLRecon3D(nn.Module):
    def __init__(self, in_channel=64, out_channel=3):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.block1 = nn.Sequential(nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=3, stride=1, padding='same'),
                                    nn.ReLU())

        self.block2 = nn.Sequential(nn.Conv3d(in_channels=in_channel//2, out_channels=in_channel//2, kernel_size=3, stride=1, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=in_channel//2, out_channels=in_channel//4, kernel_size=3, stride=1, padding='same'),
                                    nn.ReLU())

        self.block3 = nn.Sequential(nn.Conv3d(in_channels=in_channel//4, out_channels=in_channel//8, kernel_size=3, stride=1, padding='same'),
                                    nn.ReLU())

        self.block4 = nn.Sequential(nn.Conv3d(in_channels=in_channel//8, out_channels=out_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.ReLU())
    
    def forward(self, x):
        y = self.block1(x)
        y = F.interpolate(y, scale_factor=(1,2,2))

        y = self.block2(y)
        y = F.interpolate(y, scale_factor=(1,2,2))

        y = self.block3(y)
        y = F.interpolate(y, scale_factor=(1,2,2))

        y = self.block4(y)
        y = F.interpolate(y, scale_factor=(1,2,2))
        return y

class SSMTLModel(nn.Module):
    def __init__(self, in_channel=32, out_channel=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.backbone = CNN3D(in_channel=in_channel, out_channel=out_channel)

        self.arrow_head = nn.Sequential(nn.Conv2d(in_channels=out_channel, out_channels=32, kernel_size=3, padding='same'),
                                        nn.BatchNorm2d(num_features=32),
                                        nn.ReLU(),
                                        nn.Dropout2d(p=0.3),
                                        nn.MaxPool2d(2,2),
                                        nn.Flatten(),
                                        nn.Linear(128,2))
        
        self.motion_head = nn.Sequential(nn.Conv2d(in_channels=out_channel, out_channels=32, kernel_size=3, padding='same'),
                                        nn.BatchNorm2d(num_features=32),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.MaxPool2d(2,2),
                                        nn.Flatten(),
                                        nn.Linear(128,2))
        
        self.distil_head = nn.Sequential(nn.Conv2d(in_channels=out_channel, out_channels=32, kernel_size=3, padding='same'),
                                        nn.BatchNorm2d(num_features=32),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.MaxPool2d(2,2),
                                        nn.Flatten(),
                                        nn.Linear(128,1080),
                                        nn.ReLU())

        self.recon_head = SSMTLRecon(in_channel=out_channel)

    def forward(self, x_arrow, x_motion, x_recon, x_distil):
        # input_arrow -> (B,3,6,H,W)
        # input_motion -> (B,3,7,H,W)
        # input_recon -> (B,3,6,H,W)
        # input_distil -> (B,3,7,H,W)

        y_arrow = self.backbone(x_arrow)
        y_arrow = torch.squeeze(y_arrow, dim=-3)
        y_arrow = self.arrow_head(y_arrow)

        y_motion = self.backbone(x_motion)
        y_motion = torch.squeeze(y_motion, dim=-3)
        y_motion = self.motion_head(y_motion)

        y_recon = self.backbone(x_recon)
        y_recon = torch.squeeze(y_recon, dim=-3)
        y_recon = self.recon_head(y_recon)

        y_distil = self.backbone(x_distil)
        y_distil = torch.squeeze(y_distil, dim=-3)
        y_distil = self.distil_head(y_distil)

        return y_arrow, y_motion, y_recon, y_distil 

class SSMTLAutoEncoder(nn.Module):
    def __init__(self, in_channel:int=32, out_channel:int=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.encoder = CNN3D(in_channel=in_channel, out_channel=out_channel)

        self.decoder = SSMTLRecon3D(in_channel=out_channel, out_channel=3)
        
    def forward(self, x):
        emb = self.encoder(x)
        recon = self.decoder(emb)

        return recon
    
class SSMTLAutoEncArrow(nn.Module):
    def __init__(self, in_channel:int=32, out_channel:int=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.encoder = CNN3D(in_channel=in_channel, out_channel=out_channel)
        self.decoder = SSMTLRecon3D(in_channel=out_channel, out_channel=3)

        self.pool_recon = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.pool_arrow = nn.MaxPool3d(kernel_size=(7,2,2), stride=(7,2,2))
        self.arrow_head = nn.Sequential(nn.Conv2d(in_channels=out_channel, out_channels=32, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2,2),
                                        nn.Flatten(),
                                        nn.Linear(32,2))
        
    def forward(self, x_masked, x_arrow):
        emb = self.encoder(x_masked)
        emb = self.pool_recon(emb)
        y_recon = self.decoder(emb)
        
        y_arrow = self.encoder(x_arrow)
        y_arrow = self.pool_arrow(y_arrow)
        y_arrow = torch.squeeze(y_arrow, dim=-3)
        y_arrow = self.arrow_head(y_arrow)

        return y_recon, y_arrow
    
class CNN3DResReconSSMTLDec(nn.Module):
    def __init__(self, in_channel=32, out_channel=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.encoder = CNN3DRes(in_channel=in_channel, out_channel=out_channel)
        self.decoder = SSMTLRecon3D(in_channel=out_channel, out_channel=3)

    def forward(self, x):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)

        return recon