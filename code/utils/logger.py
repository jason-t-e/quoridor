import logging
import os
from datetime import datetime
import json

class QuoridorLogger:
    def __init__(self, config):
        self.config = config
        
        # Setup directories
        self.log_dir = config.get('logging', {}).get('log_dir', 'data/logs')
        self.exp_dir = config.get('logging', {}).get('experiments_dir', 'data/experiments')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Setup basic Python logger
        log_level_str = config.get('logging', {}).get('level', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        self.logger = logging.getLogger("QuoridorAI")
        self.logger.setLevel(log_level)
        
        # File handler
        log_file = os.path.join(self.log_dir, f"quoridor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Experiment state tracking
        self.metrics = {
            "games_played": 0,
            "win_rate": 0.0,
            "avg_game_length": 0.0,
            "training_loss": 0.0,
            "gpu_utilization": 0.0,
            "games_per_hour": 0.0
        }
    
    def log_metric(self, name, value):
        self.metrics[name] = value
        self.logger.info(f"Metric updated: {name} = {value}")
        
    def save_metrics(self, filename="metrics.json"):
        path = os.path.join(self.exp_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {path}")

def setup_logger(config):
    return QuoridorLogger(config)
