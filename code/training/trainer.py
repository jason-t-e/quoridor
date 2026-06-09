import torch
import torch.nn as nn
import torch.optim as optim

def train_step(model, optimizer, batch, device="cpu"):
    model.train()
    states, target_policies, target_values = batch
    
    states = torch.tensor(states, dtype=torch.float32).to(device)
    target_policies = torch.tensor(target_policies, dtype=torch.float32).to(device)
    target_values = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1).to(device)
    
    optimizer.zero_grad()
    
    pred_policies, pred_values = model(states)
    
    # MSE for value
    value_loss = nn.MSELoss()(pred_values, target_values)
    
    # Cross entropy for policy
    pred_log_policies = nn.LogSoftmax(dim=1)(pred_policies)
    policy_loss = -torch.sum(target_policies * pred_log_policies) / states.size(0)
    
    total_loss = value_loss + policy_loss
    total_loss.backward()
    optimizer.step()
    
    return policy_loss.item(), value_loss.item()
