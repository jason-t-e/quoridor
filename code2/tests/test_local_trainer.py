import pytest
import torch
import os
from training.experience_buffer import ExperienceBuffer
from training.local_trainer import LocalTrainer
from models.quoridor_net import QuoridorNet

def test_experience_buffer():
    buffer = ExperienceBuffer(capacity=10)
    
    state = torch.zeros(1, 10, 9, 9)
    strategy = torch.zeros(1, 16)
    policy = torch.zeros(1, 136)
    value = torch.tensor([[0.5]])
    rating = torch.tensor([[4.0]])
    
    buffer.add(state, strategy, policy, value, rating, agreement_bonus=0.2, outcome=1.0)
    
    assert len(buffer) == 1
    
    s, st, p, v, r, a, o = buffer.sample(1)
    
    assert s.shape == (1, 10, 9, 9)
    assert st.shape == (1, 16)
    assert p.shape == (1, 136)
    assert a.item() == pytest.approx(0.2)
    assert o.item() == pytest.approx(1.0)

def test_local_trainer():
    model = QuoridorNet()
    buffer = ExperienceBuffer(capacity=100)
    test_dir = "data/test_checkpoints"
    trainer = LocalTrainer(model, buffer, checkpoint_dir=test_dir)
    
    # Fill buffer with fake data
    for _ in range(5):
        buffer.add(
            state_tensor=torch.rand(1, 10, 9, 9),
            strategy_vector=torch.rand(1, 16),
            policy_target=torch.rand(1, 136),
            value_target=torch.rand(1, 1),
            rating_target=torch.rand(1, 1),
            agreement_bonus=0.5,
            outcome=1.0
        )
        
    loss, metrics = trainer.train_step(batch_size=4)
    
    assert loss is not None
    assert type(loss) == float
    
    # Save test
    trainer.save_checkpoint()
    assert os.path.exists(os.path.join(test_dir, "latest_champion.pt"))
    
    # Load test
    trainer.load_checkpoint() # Should print loaded correctly and not crash
