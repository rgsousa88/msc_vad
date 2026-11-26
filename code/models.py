import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

SEED = 122344
torch.manual_seed(SEED)
random.seed(SEED)

class ConvBlock3D(nn.Module):
    def __init__(self, in_channel=16, out_channel=32, norm_type='batch'):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if norm_type == 'batch':
            self.block = nn.Sequential(
                nn.Conv3d(in_channel, out_channel,
                        kernel_size=(3,3,3),
                        stride=(1,1,1),
                        padding='same'),
                nn.BatchNorm3d(out_channel),
                nn.ReLU())

        else:
            self.block = nn.Sequential(
                        nn.Conv3d(in_channel, out_channel,
                                kernel_size=(3,3,3),
                                stride=(1,1,1),
                                padding=(1,1,1)),
                        nn.InstanceNorm3d(out_channel),
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
    def __init__(self, inner_channel=16, out_channel=32, temp_pool=1, norm_type='batch'):
        super().__init__()
        self.inner_channel = inner_channel
        self.out_channel = out_channel
        self.temp_pool = temp_pool

        self.block1 = nn.Sequential(ConvBlock3D(in_channel=3, out_channel=self.inner_channel, norm_type=norm_type),
                                    ConvBlock3D(in_channel=self.inner_channel, out_channel=self.inner_channel, norm_type=norm_type),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))

        self.block2 = nn.Sequential(ConvBlock3D(in_channel=self.inner_channel, out_channel=self.out_channel, norm_type=norm_type),
                                    ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel, norm_type=norm_type),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))

        self.block3 = nn.Sequential(ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel, norm_type=norm_type),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
                                    ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel, norm_type=norm_type))
        

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = F.max_pool3d(y, kernel_size=(y.shape[2],2,2), stride=2)

        return y

class EfficientChannelAttention3D(nn.Module):
    def __init__(self, C: int, gamma: int = 2, b: int = 1):
        super().__init__()

        self.gamma = gamma
        self.b = b
        self.C = C
        self.t = int(abs((np.log2(C) + self.b) / self.gamma))
        self.k = self.t if self.t % 2 else self.t + 1

        # 1D conv over the channel descriptor (same as 2D version)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k, padding=int(self.k / 2), bias=False)

        # Adaptive pooling over the three spatial dims (D, H, W)
        # self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        # x: (B, C, D, H, W)
        y = self.avg_pool(x)                       # (B, C, T, 1, 1)
        
        # y = y.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)
        # y = y.unsqueeze(1)                         # (B, 1, C)  -> treat C as the "length" dim for Conv1d
        # y = self.conv(y)                           # (B, 1, C)
        # y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        # y = torch.sigmoid(y)

        y = y.amax(dim=2)                    # (B, C, 1, 1) — max over T, not mean
        y = y.flatten(1).unsqueeze(1)        # (B, 1, C)
        y = self.conv(y)                     # (B, 1, C)
        y = torch.sigmoid(y)
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1, 1)

        return x * y  # broadcast attention over (D, H, W)

class ChannelAttention3D(nn.Module):
    """SE-style channel attention that keeps the time axis.

    The previous ECA block aggregated over H, W *and* T (``amax(dim=2)``) before
    the gate, so every frame shared a single channel gate. Here the descriptor
    is pooled over H, W only, so the 1x1x1 MLP emits one gate per frame and the
    temporal structure is preserved.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (N, C, T, H, W)
        avg = x.mean(dim=(3, 4), keepdim=True)    # (N, C, T, 1, 1)
        mx = x.amax(dim=(3, 4), keepdim=True)     # (N, C, T, 1, 1)
        gate = torch.sigmoid(self.mlp(avg + mx))
        return x * gate

class SpatialAttention3D(nn.Module):
    """CBAM-style spatial attention computed per frame (keeps T)."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size),
                              padding=(0, padding, padding), bias=False)

    def forward(self, x):
        # x: (N, C, T, H, W)
        avg = x.mean(dim=1, keepdim=True)         # (N, 1, T, H, W)
        mx = x.amax(dim=1, keepdim=True)          # (N, 1, T, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class CBAM3D(nn.Module):
    """Channel + spatial attention for 3D feature maps, preserving the time axis.

    Replaces EfficientChannelAttention3D, which collapsed T and shared one gate
    across all frames.
    """
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention3D(channels, reduction=reduction)
        self.spatial = SpatialAttention3D(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

class CNN3DEncoder(CNN3D):
    def __init__(self, inner_channel=32, out_channel=64, norm_type='instance'):
        super().__init__(inner_channel=inner_channel, out_channel=out_channel, norm_type=norm_type)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(y1)
        y3 = self.block3(y2)
        y = self.pool1(y3)

        return y

class CNN3DEncoderAttention(CNN3D):
    def __init__(self, inner_channel=32, out_channel=64, norm_type='instance'):
        super().__init__(inner_channel=inner_channel, out_channel=out_channel, norm_type=norm_type)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.att1 = CBAM3D(channels=self.inner_channel)
        self.att2 = CBAM3D(channels=self.out_channel)
        self.att3 = CBAM3D(channels=self.out_channel)

    def forward(self, x):
        y1 = self.block1(x)
        y1 = self.att1(y1)

        y2 = self.block2(y1)
        y2 = self.att2(y2)

        y3 = self.block3(y2)
        y3 = self.att3(y3)

        y = self.pool1(y3)

        return y

class STEncoder(nn.Module):
    def __init__(self, inner_channel=32, out_channel=64, norm_type='instance'):
        super().__init__()
        self.inner_channel = inner_channel
        self.out_channel = out_channel

        self.block1 = nn.Sequential(ConvBlock3D(in_channel=3, out_channel=self.inner_channel, norm_type=norm_type),
                                    ConvBlock3D(in_channel=self.inner_channel, out_channel=self.inner_channel, norm_type=norm_type),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))

        self.block2 = nn.Sequential(ConvBlock3D(in_channel=self.inner_channel, out_channel=self.out_channel, norm_type=norm_type),
                                    ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel, norm_type=norm_type),
                                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))

        self.block3 = nn.Sequential(ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel, norm_type=norm_type),
                                    ConvBlock3D(in_channel=self.out_channel, out_channel=self.out_channel, norm_type=norm_type))
        

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(y1)
        y = self.block3(y2)

        return y

class CNN3DResSkip(CNN3D):
    def __init__(self, in_channel=32, out_channel=64, temp_pool=1):
        super().__init__(inner_channel=in_channel, out_channel=out_channel, temp_pool=temp_pool)
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
        super().__init__(inner_channel=in_channel, out_channel=out_channel, temp_pool=temp_pool)
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
        self.encoder = CNN3DEncoder(inner_channel=in_channel, out_channel=out_channel)

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
        
        self.encoder = CNN3DEncoder(inner_channel=in_channel, out_channel=out_channel, temp_pool=1)

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
    
class CNN3DDecoder(nn.Module):
    def __init__(self, inner_channel=64, out_channel=3, norm_type='instance'):
        super().__init__()
        self.in_channel = inner_channel
        self.inter_channel = inner_channel // 2
        self.out_channel = out_channel

        self.scale_factor = (1,2,2)

        def norm(c):
            return nn.BatchNorm3d(num_features=c) if norm_type == 'batch' else nn.InstanceNorm3d(num_features=c)

        self.block1 = nn.Sequential(nn.Conv3d(in_channels=inner_channel, out_channels=self.inter_channel, kernel_size=3, stride=1, padding='same'),
                                    norm(self.inter_channel),
                                    nn.ReLU())

        self.block2 = nn.Sequential(nn.Conv3d(in_channels=self.inter_channel, out_channels=self.inter_channel, kernel_size=3, stride=1, padding='same'),
                                    norm(self.inter_channel),
                                    nn.ReLU())

        self.block3 = nn.Sequential(nn.Conv3d(in_channels=self.inter_channel, out_channels=self.inter_channel, kernel_size=3, stride=1, padding='same'),
                                    norm(self.inter_channel),
                                    nn.ReLU())

        self.block4 = nn.Conv3d(self.inter_channel, self.out_channel, kernel_size=3, stride=1, padding='same')
    
    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.scale_factor)
        y = self.block1(y)

        y = F.interpolate(y, scale_factor=self.scale_factor)
        y = self.block2(y)

        y = F.interpolate(y, scale_factor=self.scale_factor)
        y = self.block3(y)
        
        y = F.interpolate(y, scale_factor=self.scale_factor)
        y = self.block4(y)
        y = y.sigmoid()
        
        return y

class STDecoder(nn.Module):
    def __init__(self, inner_channel=64, out_channel=3):
        super().__init__()
        self.in_channel = inner_channel
        self.inter_channel = inner_channel // 2
        self.out_channel = out_channel

        self.scale_factor = (1,2,2)

        self.block1 = nn.Sequential(nn.Conv3d(in_channels=inner_channel, out_channels=self.inter_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.InstanceNorm3d(num_features=self.inter_channel),
                                    nn.ReLU())

        self.block2 = nn.Sequential(nn.Conv3d(in_channels=self.inter_channel, out_channels=self.inter_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.InstanceNorm3d(num_features=self.inter_channel),
                                    nn.ReLU())

        self.block3 = nn.Sequential(nn.Conv3d(in_channels=self.inter_channel, out_channels=out_channel, kernel_size=3, stride=1, padding='same'),
                                    nn.InstanceNorm3d(num_features=out_channel),
                                    nn.ReLU())
    
    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.scale_factor)
        y = self.block1(y)

        y = F.interpolate(y, scale_factor=self.scale_factor)
        y = self.block2(y)

        y = F.interpolate(y, scale_factor=self.scale_factor)
        y = self.block3(y)
        y = y.sigmoid()
        
        return y

class SSMTLModel(nn.Module):
    def __init__(self, in_channel=32, out_channel=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.backbone = CNN3D(inner_channel=in_channel, out_channel=out_channel)

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
        self.encoder = CNN3D(inner_channel=in_channel, out_channel=out_channel)

        self.decoder = CNN3DDecoder(inner_channel=out_channel, out_channel=3)
        
    def forward(self, x):
        emb = self.encoder(x)
        recon = self.decoder(emb)

        return recon
    
class SSMTLAutoEncArrow(nn.Module):
    def __init__(self, in_channel:int=32, out_channel:int=64):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.encoder = CNN3D(inner_channel=in_channel, out_channel=out_channel)
        self.decoder = CNN3DDecoder(inner_channel=out_channel, out_channel=3)

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
    
class CNN3DReconBase(nn.Module):
    def __init__(self, inner_channel=32, out_channel=64):
        super().__init__()
        self.inner_channel = inner_channel
        self.out_channel = out_channel

        self.encoder = CNN3DEncoder(inner_channel=inner_channel, out_channel=out_channel)
        self.decoder = CNN3DDecoder(inner_channel=out_channel, out_channel=3)

    def forward(self, x):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)

        return recon
    
class CNN3DReconArrow(nn.Module):
    def __init__(self, inner_channel=32, out_channel=64):
        super().__init__()
        self.inner_channel = inner_channel
        self.out_channel = out_channel

        self.encoder = CNN3DEncoder(inner_channel=inner_channel, out_channel=out_channel)
        self.decoder = CNN3DDecoder(inner_channel=out_channel, out_channel=3)
        self.pool_arrow = nn.MaxPool3d(kernel_size=(7,2,2), stride=(7,2,2))
        self.arrow_head = nn.Sequential(nn.Conv2d(in_channels=out_channel, out_channels=32, kernel_size=3, padding='same'),
                                        nn.BatchNorm2d(num_features=32),
                                        nn.ReLU(),
                                        nn.Dropout2d(p=0.3),
                                        nn.MaxPool2d(2,2),
                                        nn.Flatten(),
                                        nn.Linear(32,2))

    def forward(self, x, x_arrow):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)

        enc_arrow = self.encoder(x_arrow)
        #enc_arrow = self.pool_arrow(enc_arrow)
        enc_arrow = F.max_pool3d(enc_arrow, kernel_size=(enc_arrow.shape[2],2,2), stride=(enc_arrow.shape[2],2,2))
        enc_arrow = torch.squeeze(enc_arrow, dim=-3)
        arrow_pred = self.arrow_head(enc_arrow)

        return recon, arrow_pred

class TemporalOrderHead(nn.Module):
    """Order-aware head for the temporal-shuffle ("arrow"/time) pretask.

    The previous heads max/avg-pooled the encoder features over the time axis and
    then classified, which is order-invariant: shuffling the frames only permutes
    the pooled inputs, so the prediction barely changed. Here we keep one token
    per frame, tag each with a learned positional embedding, run a lightweight
    Transformer encoder over the time axis, and only then pool over frames. The
    pooled representation now depends on the frame order, which is what the task
    asks the model to detect.
    """
    def __init__(self, in_channels: int, n_classes: int = 4, max_frames: int = 33,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 1,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, feat):
        # feat: (N, C, T, H, W)
        x = feat.mean(dim=(3, 4))                 # (N, C, T) - descriptor per frame
        x = x.transpose(1, 2)                     # (N, T, C)
        x = self.proj(x)                          # (N, T, d_model)

        T = x.shape[1]
        if T <= self.pos_emb.shape[1]:
            pe = self.pos_emb[:, :T]
        else:
            pe = F.interpolate(self.pos_emb.transpose(1, 2), size=T,
                               mode='linear', align_corners=False).transpose(1, 2)
        x = x + pe.to(x.dtype)

        x = self.transformer(x)                   # (N, T, d_model)
        x = self.norm(x).mean(dim=1)              # (N, d_model)
        return self.classifier(x)


class CNN3DReconTime(nn.Module):
    def __init__(self, inner_channel=32, out_channel=64):
        super().__init__()
        self.inner_channel = inner_channel
        self.out_channel = out_channel

        self.encoder = CNN3DEncoder(inner_channel=inner_channel, out_channel=out_channel)
        self.decoder = CNN3DDecoder(inner_channel=out_channel, out_channel=3)
        self.time_head = TemporalOrderHead(in_channels=out_channel, n_classes=4)

    def forward(self, x, x_arrow):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)

        # Keep the full (N, C, T, H, W) feature map: TemporalOrderHead models the
        # temporal order instead of collapsing T with a max-pool.
        enc_time = self.encoder(x_arrow)
        time_pred = self.time_head(enc_time)

        return recon, time_pred

class CNN3DReconTimeAttention(CNN3DReconTime):
    def __init__(self, inner_channel=32, out_channel=64):
        super().__init__(inner_channel=inner_channel, out_channel=out_channel)

        self.encoder = CNN3DEncoderAttention(inner_channel=inner_channel, out_channel=out_channel)
        # time_head is inherited from CNN3DReconTime (TemporalOrderHead); the
        # channel attention is provided by the encoder above.

class STReconTime(nn.Module):
    def __init__(self, inner_channel=32, out_channel=64):
        super().__init__()
        self.inner_channel = inner_channel
        self.out_channel = out_channel

        self.encoder = STEncoder(inner_channel=inner_channel, out_channel=out_channel, norm_type='instance')
        self.decoder = STDecoder(inner_channel=out_channel, out_channel=3)

        self.time_head = TemporalOrderHead(in_channels=out_channel, n_classes=4)

    def forward(self, x, x_arrow):
        encoded = self.encoder(x)
        encoded = F.max_pool3d(encoded, kernel_size=(1,2,2), stride=(1,2,2))
        recon = self.decoder(encoded)

        # Order-aware temporal head over the full feature map (keeps T).
        enc_time = self.encoder(x_arrow)
        time_pred = self.time_head(enc_time)

        return recon, time_pred
