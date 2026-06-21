import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        return out + identity
