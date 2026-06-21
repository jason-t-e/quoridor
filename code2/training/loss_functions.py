import torch
import torch.nn.functional as F

def total_loss(batch_policy_targets, batch_value_targets, batch_rating_targets, 
               model_policy_pred, model_value_pred, model_rating_pred,
               agreement_bonuses, outcomes, config):
    """
    Computes the 5-component loss function from the TRD.
    """
    # 1. POLICY
    L_policy = -torch.sum(batch_policy_targets * torch.log(model_policy_pred + 1e-8), dim=1).mean()
    
    # 2. VALUE
    L_value = F.mse_loss(model_value_pred, batch_value_targets)
    
    # 3. RATING
    L_rating = F.mse_loss(model_rating_pred, batch_rating_targets)
    L_rating_max = F.relu(3.0 - model_rating_pred.mean())
    
    # 4. STRATEGY
    L_strategy = -agreement_bonuses.mean()
    
    # 5. DEFEAT
    defeat_mask = (outcomes == -1).float()
    L_defeat = (defeat_mask * F.mse_loss(model_value_pred, batch_value_targets, reduction='none')).mean()
    
    total = (config['w_policy'] * L_policy +
             config['w_value'] * L_value +
             config['w_rating'] * L_rating +
             config['w_rating_max'] * L_rating_max +
             config['w_strategy'] * L_strategy +
             config['w_defeat'] * L_defeat)
             
    return total, {
        "policy": L_policy.item(),
        "value": L_value.item(),
        "rating": L_rating.item(),
        "strategy": L_strategy.item(),
        "defeat": L_defeat.item()
    }
