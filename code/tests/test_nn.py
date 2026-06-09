import torch
from models.quoridor_net import QuoridorNet

def test_quoridor_net_shapes():
    model = QuoridorNet()
    # Batch size of 2
    dummy_input = torch.zeros((2, 10, 9, 9))
    policy_logits, value = model(dummy_input)
    
    assert policy_logits.shape == (2, 136)
    assert value.shape == (2, 1)
    
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)
