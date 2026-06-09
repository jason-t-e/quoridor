import random
from collections import deque
import numpy as np

class ExperienceBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state_tensor, policy_probs, value):
        self.buffer.append((state_tensor, policy_probs, value))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return np.array(states), np.array(policies), np.array(values)
        
    def __len__(self):
        return len(self.buffer)
