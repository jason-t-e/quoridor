import os
from collections import deque
import random
import torch

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state_tensor, strategy_vector, policy_target, value_target, rating_target, agreement_bonus, outcome):
        # We need to save the outcome later once the game ends, so we temporarily
        # store experiences without outcome, then update them.
        self.buffer.append({
            'state': state_tensor,
            'strategy': strategy_vector,
            'policy': policy_target,
            'value': value_target,
            'rating': rating_target,
            'agreement': agreement_bonus,
            'outcome': outcome
        })
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
            
        states = torch.cat([b['state'] for b in batch], dim=0)
        strategies = torch.cat([b['strategy'] for b in batch], dim=0)
        policies = torch.cat([b['policy'] for b in batch], dim=0)
        values = torch.cat([b['value'] for b in batch], dim=0)
        ratings = torch.cat([b['rating'] for b in batch], dim=0)
        agreements = torch.tensor([b['agreement'] for b in batch], dtype=torch.float32)
        outcomes = torch.tensor([b['outcome'] for b in batch], dtype=torch.float32)
        
        return states, strategies, policies, values, ratings, agreements, outcomes

    def __len__(self):
        return len(self.buffer)
