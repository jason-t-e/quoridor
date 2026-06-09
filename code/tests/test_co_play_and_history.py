import os
import shutil
import tempfile
import torch
import numpy as np
from engine.game import QuoridorGame
from engine.moves import PawnMove
from models.quoridor_net import QuoridorNet, UniformModel
from training.self_play_worker import play_co_play_game
from utils.history_tracker import HistoryTracker

def test_history_tracker():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test_history.json")
    try:
        tracker = HistoryTracker(filepath=temp_file)
        game = QuoridorGame()
        
        # Test key generation
        key = tracker.get_state_key(game.board)
        assert "P0:0_4|P1:8_4" in key
        
        # Record a step
        board_before = game.board.clone()
        move = PawnMove(1, 4)
        game.apply_move(move)
        board_after = game.board.clone()
        
        tracker.record_game_step(board_before, move, board_after)
        key_before = tracker.get_state_key(board_before)
        key_after = tracker.get_state_key(board_after)
        
        assert key_before in tracker.graph
        assert key_after in tracker.graph
        assert tracker.graph[key_before]["visit_count"] == 1
        assert "Pawn(1, 4)" in tracker.graph[key_before]["moves"]
        
        # Record game outcome
        tracker.record_game_outcome([key_before, key_after], 0)
        assert tracker.graph[key_before]["winner_counts"]["0"] == 1
        
        # Save and reload
        tracker.save()
        assert os.path.exists(temp_file)
        
        tracker2 = HistoryTracker(filepath=temp_file)
        assert key_before in tracker2.graph
        assert tracker2.graph[key_before]["visit_count"] == 1
        
    finally:
        shutil.rmtree(temp_dir)

def test_play_co_play_game_alphazero_vs_minimax():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = QuoridorNet().to(device)
    model2 = UniformModel().to(device)  # Dummy model representing Minimax
    
    # Pit alphazero (model 1) vs minimax (depth 2)
    m1_data, m2_data, transitions, winner = play_co_play_game(
        "alphazero", model1, "minimax", model2, mcts_simulations=2, model1_is_player_0=True, minimax_depth=2
    )
    
    assert len(transitions) > 0
    assert winner in (0, 1, None)
    
    # Check data format for AlphaZero player
    if m1_data:
        state, policy, value = m1_data[0]
        assert state.shape == (10, 9, 9)
        assert len(policy) == 136
        assert value in (1.0, -1.0, 0.0)

def test_play_co_play_game_uniform_vs_minimax():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = UniformModel().to(device)
    model2 = UniformModel().to(device)
    
    # Pit MCTS Uniform vs Minimax
    m1_data, m2_data, transitions, winner = play_co_play_game(
        "mcts_uniform", model1, "minimax", model2, mcts_simulations=2, model1_is_player_0=False, minimax_depth=2
    )
    
    assert len(transitions) > 0
    assert winner in (0, 1, None)
