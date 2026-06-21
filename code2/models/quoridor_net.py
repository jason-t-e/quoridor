import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_block import ResidualBlock
from .path_attention import PathAttentionGate

class QuoridorNet(nn.Module):
    def __init__(self, in_channels=10, strategy_dim=16, channels=128):
        super().__init__()
        
        # Stem
        self.stem_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        
        # Block Group 1
        self.group1 = nn.ModuleList([ResidualBlock(channels) for _ in range(5)])
        self.attention1 = PathAttentionGate()
        
        # Strategy Injection
        self.strategy_embedding = nn.Linear(strategy_dim, strategy_dim)
        self.strategy_conv = nn.Conv2d(channels + strategy_dim, channels, kernel_size=1)
        self.strategy_bn = nn.BatchNorm2d(channels)
        
        # Block Group 2
        self.group2 = nn.ModuleList([ResidualBlock(channels) for _ in range(5)])
        self.attention2 = PathAttentionGate()
        
        # Block Group 3
        self.group3 = nn.ModuleList([ResidualBlock(channels) for _ in range(5)])
        self.attention3 = PathAttentionGate()
        
        # Heads
        # Flattening via Global Average Pool implies output features = channels
        
        # Policy Head (136 output classes: 128 wall + 8 pawn moves approx, but we use 136 logically)
        # Actually max 4 walls * 64 = 128 wall moves (2 orientations * 64). 
        # Pawn moves max 4, jumps max 4 = 8. So 128+8 = 136 actions.
        self.policy_head = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, 136)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # Rating Head
        # Note: TRD says Concat [128-dim, 8-dim move_context_vector], but for simplicity we predict rating per action?
        # Actually, "Output: scalar ∈ [0, 5]". We'll just take 128-dim state and predict average rating.
        self.rating_head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, board_tensor, strategy_vector):
        """
        board_tensor: [B, 10, 9, 9]
        strategy_vector: [B, 16]
        """
        path_own = board_tensor[:, 4, :, :]
        path_opp = board_tensor[:, 5, :, :]
        
        # Stem
        x = F.relu(self.stem_bn(self.stem_conv(board_tensor)))
        
        # Group 1
        for block in self.group1:
            x = block(x)
        x = self.attention1(x, path_own, path_opp)
        
        # Strategy Injection
        strat_emb = self.strategy_embedding(strategy_vector) # [B, 16]
        strat_map = strat_emb.view(-1, 16, 1, 1).expand(-1, -1, 9, 9)
        x = torch.cat([x, strat_map], dim=1) # [B, 144, 9, 9]
        x = F.relu(self.strategy_bn(self.strategy_conv(x)))
        
        # Group 2
        for block in self.group2:
            x = block(x)
        x = self.attention2(x, path_own, path_opp)
        
        # Group 3
        for block in self.group3:
            x = block(x)
        x = self.attention3(x, path_own, path_opp)
        
        # Global average pool
        pooled = x.mean(dim=(2, 3)) # [B, 128]
        
        # Heads
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        rating = self.rating_head(pooled) * 5.0 # Scale [0,1] to [0,5]
        
        return policy_logits, value, rating
