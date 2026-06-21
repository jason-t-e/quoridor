import os
import torch
import torch.optim as optim
from .experience_buffer import ExperienceBuffer
from models.quoridor_net import QuoridorNet
from training.loss_functions import total_loss

class LocalTrainer:
    def __init__(self, model: QuoridorNet, buffer: ExperienceBuffer, lr=1e-4, checkpoint_dir="data/checkpoints"):
        self.model = model
        self.buffer = buffer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "latest_champion.pt")
        
        # Training config matching the TRD
        self.loss_config = {
            'w_policy': 1.0,
            'w_value': 1.0,
            'w_rating': 1.0,
            'w_rating_max': 0.1,
            'w_strategy': 0.4,
            'w_defeat': 1.5
        }

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location='cpu'))
            print(f"Loaded checkpoint from {self.checkpoint_path}")
        else:
            print("No checkpoint found. Starting fresh.")

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            print("Not enough experiences to train yet.")
            return

        self.model.train()
        
        states, strategies, policies, values, ratings, agreements, outcomes = self.buffer.sample(batch_size)
        
        self.optimizer.zero_grad()
        
        model_policy, model_value, model_rating = self.model(states, strategies)
        
        loss, metrics = total_loss(
            policies, values, ratings,
            model_policy, model_value, model_rating,
            agreements, outcomes, self.loss_config
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), metrics
