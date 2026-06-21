import pytest
import torch
import numpy as np

from game_engine.board import BoardState, board_to_tensor
from models.quoridor_net import QuoridorNet
from strategy.strategy_guide import StrategyGuide
from search.transposition_table import MCTSTranspositionTable, SolverTranspositionTable
from search.mcts import MCTS
from search.endgame_solver import EndgameSolver
from training.loss_functions import total_loss

def test_quoridor_net_forward():
    net = QuoridorNet()
    board = BoardState()
    tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0)
    strat = torch.zeros((1, 16), dtype=torch.float32)
    
    policy_logits, value, rating = net(tensor, strat)
    
    assert policy_logits.shape == (1, 136)
    assert value.shape == (1, 1)
    assert rating.shape == (1, 1)
    
    assert -1.0 <= value.item() <= 1.0
    assert 0.0 <= rating.item() <= 5.0

def test_mcts_search():
    board = BoardState()
    net = QuoridorNet()
    guide = StrategyGuide()
    tt = MCTSTranspositionTable()
    mcts = MCTS()
    
    # 50ms time budget
    moves, probs = mcts.search(board, net, guide, tt, time_budget_ms=50)
    
    assert len(moves) > 0
    assert len(probs) == len(moves)
    assert np.isclose(sum(probs), 1.0)
    
def test_endgame_solver():
    board = BoardState()
    # Endgame is when total_walls <= 4
    board.walls_remaining = {0: 1, 1: 1}
    # Move pawns to near the goal to speed up solver
    board.pawn_positions = {0: (7, 4), 1: (1, 4)}
    
    solver = EndgameSolver()
    tt = SolverTranspositionTable()
    
    # Run solver
    move = solver.solve(board, tt)
    assert move is not None

def test_loss_function():
    batch_size = 4
    policy_targets = torch.ones((batch_size, 136)) / 136.0
    value_targets = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
    rating_targets = torch.tensor([[4.0], [2.0], [3.5], [1.0]])
    
    model_policy_pred = torch.ones((batch_size, 136)) / 136.0
    model_value_pred = torch.tensor([[0.8], [-0.9], [0.5], [-0.5]])
    model_rating_pred = torch.tensor([[3.8], [2.1], [3.0], [1.5]])
    
    agreement_bonuses = torch.tensor([0.5, 0.0, 0.8, 0.2])
    outcomes = torch.tensor([1, -1, 1, -1])
    
    config = {
        'w_policy': 1.0,
        'w_value': 1.0,
        'w_rating': 1.0,
        'w_rating_max': 0.1,
        'w_strategy': 0.4,
        'w_defeat': 1.5
    }
    
    loss, metrics = total_loss(policy_targets, value_targets, rating_targets,
                               model_policy_pred, model_value_pred, model_rating_pred,
                               agreement_bonuses, outcomes, config)
                               
    assert loss.item() > 0
    assert metrics['strategy'] <= 0 # Strategy bonus should be negative
