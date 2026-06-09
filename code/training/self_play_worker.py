import torch
import numpy as np
from search.mcts import MCTS
from engine.game import QuoridorGame
from models.board_encoder import encode_board, decode_action

def play_self_play_game(model, mcts_simulations=800):
    model.eval()
    mcts = MCTS(model, num_simulations=mcts_simulations)
    game = QuoridorGame()
    
    states = []
    policies = []
    players = []
    
    while not game.is_terminal:
        action_probs = mcts.search(game.board)
        
        # Action space size is 136
        policy = np.zeros(136, dtype=np.float32)
        for a, p in action_probs.items():
            policy[a] = p
            
        states.append(encode_board(game.board))
        policies.append(policy)
        players.append(game.board.current_player)
        
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        chosen_action = np.random.choice(actions, p=probs)
        
        move = decode_action(game.board, chosen_action)
        game.apply_move(move)
        
    winner = game.winner
    
    values = []
    for p in players:
        if p == winner:
            values.append(1.0)
        else:
            values.append(-1.0)
            
    return list(zip(states, policies, values))

def play_co_play_game(agent_type1, model1, agent_type2, model2, mcts_simulations=800, model1_is_player_0=True, minimax_depth=8):
    if model1_is_player_0:
        p0_type, p0_model = agent_type1, model1
        p1_type, p1_model = agent_type2, model2
    else:
        p0_type, p0_model = agent_type2, model2
        p1_type, p1_model = agent_type1, model1
        
    game = QuoridorGame()
    
    p0_mcts = None
    p1_mcts = None
    
    if p0_type != "minimax" and p0_model is not None:
        p0_model.eval()
        p0_mcts = MCTS(p0_model, num_simulations=mcts_simulations)
    if p1_type != "minimax" and p1_model is not None:
        p1_model.eval()
        p1_mcts = MCTS(p1_model, num_simulations=mcts_simulations)
        
    game_history = []
    transitions = []
    
    while not game.is_terminal:
        curr_p = game.board.current_player
        curr_type = p0_type if curr_p == 0 else p1_type
        
        board_before = game.board.clone()
        
        if curr_type == "minimax":
            import math
            from search.minimax import alphabeta
            from engine.rules import get_all_legal_moves
            _, move = alphabeta(game.board, depth=minimax_depth, alpha=-math.inf, beta=math.inf, maximizing_player=True, root_player=curr_p)
            if move is None:
                import random
                move = random.choice(get_all_legal_moves(game.board))
                
            policy = np.zeros(136, dtype=np.float32)
            from models.board_encoder import encode_action
            try:
                action_idx = encode_action(board_before, move)
                policy[action_idx] = 1.0
            except:
                pass
        else:
            mcts = p0_mcts if curr_p == 0 else p1_mcts
            action_probs = mcts.search(game.board)
            policy = np.zeros(136, dtype=np.float32)
            for a, p in action_probs.items():
                policy[a] = p
                
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            chosen_action = np.random.choice(actions, p=probs)
            move = decode_action(game.board, chosen_action)
            
        game_history.append((encode_board(board_before), policy, curr_p))
        game.apply_move(move)
        board_after = game.board.clone()
        
        transitions.append((board_before, move, board_after))
        
    winner = game.winner
    
    model1_data = []
    model2_data = []
    
    for state, policy, p in game_history:
        if winner is None:
            value = 0.0
        else:
            value = 1.0 if p == winner else -1.0
            
        is_model1 = (p == 0 and model1_is_player_0) or (p == 1 and not model1_is_player_0)
        if is_model1:
            model1_data.append((state, policy, value))
        else:
            model2_data.append((state, policy, value))
            
    return model1_data, model2_data, transitions, winner
