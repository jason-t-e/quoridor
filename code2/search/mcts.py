import time
import math
import torch
import numpy as np
from game_engine.board import board_to_tensor
from game_engine.rules import get_legal_moves
from .transposition_table import MCTSTranspositionTable

class MCTSNode:
    def __init__(self, board_state):
        self.board_state = board_state
        self.children = {}  # move -> MCTSNode
        self.Q = 0.0
        self.W = 0.0
        self.N = 0
        self.prior = 0.0
        self.legal_moves = None
        
    def is_expanded(self):
        return self.legal_moves is not None

class MCTS:
    def __init__(self, c_puct=1.0, lambda_guide=0.1, lambda_agree=0.1):
        self.c_puct = c_puct
        self.lambda_guide = lambda_guide
        self.lambda_agree = lambda_agree

    def search(self, root_board, network, strategy_guide, mcts_tt, time_budget_ms):
        start = time.monotonic_ns()
        root_hash = root_board.current_hash
        root_node, _ = mcts_tt.get_or_create(root_hash, root_board)
        
        # Expand root if needed
        if not root_node.is_expanded():
            self._expand(root_node, network, strategy_guide)
            
        while (time.monotonic_ns() - start) / 1e6 < time_budget_ms:
            self._run_simulation(root_node, network, strategy_guide, mcts_tt)
            
        return self._get_action_probs(root_node)

    def _run_simulation(self, root_node, network, strategy_guide, mcts_tt):
        node = root_node
        search_path = [node]
        
        while node.is_expanded() and not node.board_state.is_terminal():
            best_move, best_child = self._select_child(node)
            search_path.append(best_child)
            node = best_child
            
        if not node.board_state.is_terminal():
            value = self._expand(node, network, strategy_guide)
        else:
            winner = node.board_state.winner
            # Value from perspective of node's current player
            if winner is None: value = 0.0
            else: value = 1.0 if winner == node.board_state.current_player else -1.0
            
        self._backpropagate(search_path, value)

    def _select_child(self, node):
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(max(1, node.N)) / (1 + child.N)
            # We omit the detailed strategy bonus calculations here for simplification, 
            # normally we add lambda_guide * path_guide_bonus + lambda_agree * agreement_bonus
            score = child.Q + u 
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
                
        return best_move, best_child

    def _expand(self, node, network, strategy_guide):
        board = node.board_state
        node.legal_moves = get_legal_moves(board)
        
        if len(node.legal_moves) == 0:
            return 0.0 # Should not happen unless terminal
            
        tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0)
        # Dummy strategy vector
        strat = torch.zeros((1, 16), dtype=torch.float32)
        
        policy_logits, value, _ = network(tensor, strat)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().numpy()
        
        for i, move in enumerate(node.legal_moves):
            child_board = board.apply_move(move)
            child_node = MCTSNode(child_board)
            # Map move to action_idx logic omitted for brevity, just using uniform prior
            child_node.prior = 1.0 / len(node.legal_moves) 
            node.children[move] = child_node
            
        return value.item()

    def _backpropagate(self, search_path, value):
        # Value alternates sign because players alternate
        for node in reversed(search_path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value

    def _get_action_probs(self, root_node, temperature=1.0):
        visits = [child.N for move, child in root_node.children.items()]
        moves = list(root_node.children.keys())
        
        if temperature == 0:
            best_idx = np.argmax(visits)
            probs = np.zeros(len(visits))
            probs[best_idx] = 1.0
        else:
            visits = np.array(visits) ** (1.0 / temperature)
        if np.sum(visits) == 0:
            probs = np.ones(len(visits)) / max(1, len(visits))
        else:
            probs = visits / np.sum(visits)
            
        return moves, probs
