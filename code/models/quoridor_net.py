import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        return F.relu(x)

class QuoridorNet(nn.Module):
    def __init__(self, num_channels=128, num_blocks=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(10, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 136)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 9 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

class UniformModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size = x.size(0)
        policy_logits = torch.zeros(batch_size, 136, device=x.device)
        value = torch.zeros(batch_size, 1, device=x.device)
        return policy_logits, value
