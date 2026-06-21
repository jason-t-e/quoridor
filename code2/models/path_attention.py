import torch
import torch.nn as nn

class PathAttentionGate(nn.Module):
    """
    BFS distance maps gate spatial features. attention_scale is a LEARNED parameter.
    """
    def __init__(self, attention_scale_init=0.5):
        super().__init__()
        self.attention_scale = nn.Parameter(torch.tensor(attention_scale_init))

    def forward(self, features, path_map_own, path_map_opp):
        # path maps are [batch, 9, 9] (from channels 4 and 5)
        path_diff_map = path_map_opp - path_map_own
        # We need to flatten H,W to apply softmax over spatial dimensions
        batch_size = features.size(0)
        flattened_diff = path_diff_map.view(batch_size, -1)
        attention = torch.softmax(flattened_diff, dim=1).view(batch_size, 1, 9, 9)
        
        gated = features * (1 + self.attention_scale * attention)
        return gated + features # Residual connection
