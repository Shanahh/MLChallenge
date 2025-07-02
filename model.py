import torch
import torch.nn as nn
import torch.nn.functional as F

# constants
KERNEL_SIZE_CONV =  3
KERNEL_SIZE_RESAMPLE = 2
KERNEL_SIZE_FINAL = 1
STRIDE_RESAMPLE = 2
PADDING = 1
FINAL_CHANNELS = 1

class UNet(nn.Module):
    """
    The UNet.
    """
    def __init__(self):
        super().__init__()
        # encoder blocks
        self.enc1 = EncoderBlockGeneric(4, 16)
        self.enc2 = EncoderBlockGeneric(16, 32)
        self.enc3 = EncoderBlockBottleneck(32, 64)
        # decoder blocks
        self.dec3 = DecoderBlockBottleneck(64, 32)
        self.dec2 = DecoderBlockGeneric(32, 16)
        self.dec1 = DecoderBlockFinal(16)

    def forward(self, x):
        # use encoders and save skip connections
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x = self.enc3(x)
        # use decoders and use skip connections
        x = self.dec3(x)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return x

class EncoderBlockGeneric(nn.Module):
    """
    Generic Encoder Block for the U-Net architecture with 2 convolutional layers and 1 pooling layer
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=KERNEL_SIZE_RESAMPLE, stride=STRIDE_RESAMPLE)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip

class EncoderBlockBottleneck(nn.Module):
    """
    Encoder Block at the bottleneck (last encoder block before the first decoder block) for the U-Net architecture
    Only 1 convolutional layer, no pooling
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class DecoderBlockBottleneck(nn.Module):
    """
    Decoder Block at the bottleneck (first decoder block after the last encoder block) for the U-Net architecture
    Only 1 convolutional layer, 1 up-convolutional layer
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=KERNEL_SIZE_RESAMPLE, stride=STRIDE_RESAMPLE)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.upconv(x)
        return x

class DecoderBlockGeneric(nn.Module):
    """
    Generic Decoder Block for the U-Net architecture with 2 convolutional layers and 1 up-convolutional layer
    Connects with skip connections from generic encoder blocks
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_chan_skip = 2 * in_channels
        self.conv1 = nn.Conv2d(in_chan_skip, in_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=KERNEL_SIZE_RESAMPLE, stride=STRIDE_RESAMPLE)

    def forward(self, x, skip_connection):
        x = torch.cat([x, skip_connection], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.upconv(x)
        return x

class DecoderBlockFinal(nn.Module):
    """
    Generic Decoder Block for the U-Net architecture with 2 convolutional layers
    Instead of having an up-convolutional layer, it includes the final layer with 1x1 convolution
    """
    def __init__(self, in_channels):
        super().__init__()
        in_chan_skip = 2 * in_channels
        self.conv1 = nn.Conv2d(in_chan_skip, in_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.final_conv = nn.Conv2d(in_channels, FINAL_CHANNELS, kernel_size=KERNEL_SIZE_FINAL)

    def forward(self, x, skip_connection):
        x = torch.cat([x, skip_connection], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x