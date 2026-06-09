"""
MCTS with Transposition Table (DAG) using Zobrist hashing.

The tree is stored as a DAG in self.tt (int -> MCTSNode), keyed by the
board's Zobrist hash.  This allows node reuse across different move orders
that reach the same position, and across consecutive search calls within
the same game.
"""

import math
import numpy as np
import torch
from engine.rules import get_all_legal_moves
from models.board_encoder import encode_board, encode_action, decode_action
from engine.game import QuoridorGame


class MCTSNode:
    __slots__ = ('visit_count', 'value_sum', 'is_expanded',
                 'action_priors', 'action_next_keys')

    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.action_priors = {}        # action_idx -> prior probability
        self.action_next_keys = {}     # action_idx -> zobrist hash (int)

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_probs):
        self.is_expanded = True
        self.action_priors = action_probs


class MCTS:
    def __init__(self, model, c_puct=1.5, num_simulations=800,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.last_search_depth = 0
        self.tt = {}   # zobrist hash (int) -> MCTSNode

    @torch.no_grad()
    def search(self, initial_state):
        root_key = initial_state.current_hash
        if root_key not in self.tt:
            self.tt[root_key] = MCTSNode()
        root = self.tt[root_key]

        self.last_search_depth = 0

        # Expand root if needed
        if not root.is_expanded:
            action_probs = self._evaluate_and_expand(initial_state)
            root.expand(action_probs)

        # Add Dirichlet noise to root priors
        noisy_priors = {}
        if root.action_priors:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.action_priors))
            for i, (a, prob) in enumerate(root.action_priors.items()):
                noisy_priors[a] = ((1 - self.dirichlet_eps) * prob
                                   + self.dirichlet_eps * noise[i])

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = initial_state.clone()
            terminal = False

            # --- Selection ---
            while node.is_expanded:
                best_score = -float('inf')
                best_action = None

                priors = noisy_priors if node is root else node.action_priors

                for action, prior in priors.items():
                    child_visits = 0
                    child_q = 0.0
                    if action in node.action_next_keys:
                        child_key = node.action_next_keys[action]
                        child_node = self.tt.get(child_key)
                        if child_node is not None:
                            child_visits = child_node.visit_count
                            child_q = child_node.value()

                    u = (self.c_puct * prior
                         * math.sqrt(node.visit_count) / (1 + child_visits))
                    score = child_q + u
                    if score > best_score:
                        best_score = score
                        best_action = action

                if best_action is None:
                    break

                # Apply the move
                move = decode_action(current_state, best_action)
                game = QuoridorGame()
                game.board = current_state
                game.apply_move(move)
                current_state = game.board

                # Lazily record the edge
                if best_action not in node.action_next_keys:
                    node.action_next_keys[best_action] = current_state.current_hash

                child_key = node.action_next_keys[best_action]
                if child_key not in self.tt:
                    self.tt[child_key] = MCTSNode()
                node = self.tt[child_key]
                search_path.append(node)

                if game.is_terminal:
                    terminal = True
                    break

            self.last_search_depth = max(
                self.last_search_depth, len(search_path))

            # --- Evaluation ---
            if terminal:
                value = -1.0
            else:
                value = self._expand_leaf(node, current_state)

            # --- Backpropagation ---
            for n in reversed(search_path):
                n.value_sum += value
                n.visit_count += 1
                value = -value

        # --- Collect visit counts ---
        action_visits = {}
        total = 0
        for action in root.action_priors:
            if action in root.action_next_keys:
                child_key = root.action_next_keys[action]
                child_node = self.tt.get(child_key)
                if child_node is not None:
                    action_visits[action] = child_node.visit_count
                    total += child_node.visit_count

        if total == 0:
            n = len(root.action_priors)
            return {a: 1.0 / n for a in root.action_priors}

        return {a: v / total for a, v in action_visits.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_and_expand(self, state):
        """Run the neural net on *state* and return normalised action probs."""
        device = next(self.model.parameters()).device
        board_tensor = (torch.tensor(encode_board(state))
                        .unsqueeze(0).to(device))
        policy_logits, _ = self.model(board_tensor)
        policy_probs = (torch.softmax(policy_logits, dim=1)
                        .squeeze(0).cpu().numpy())

        legal_moves = get_all_legal_moves(state)
        action_probs = {}
        for m in legal_moves:
            idx = encode_action(state, m)
            action_probs[idx] = float(policy_probs[idx])

        total = sum(action_probs.values())
        if total > 0:
            for a in action_probs:
                action_probs[a] /= total
        elif action_probs:
            n = len(action_probs)
            for a in action_probs:
                action_probs[a] = 1.0 / n
        return action_probs

    def _expand_leaf(self, node, state):
        """Expand an unexpanded leaf node.  Returns the value estimate."""
        device = next(self.model.parameters()).device
        board_tensor = (torch.tensor(encode_board(state))
                        .unsqueeze(0).to(device))
        policy_logits, value_tensor = self.model(board_tensor)
        value = value_tensor.item()
        policy_probs = (torch.softmax(policy_logits, dim=1)
                        .squeeze(0).cpu().numpy())

        legal_moves = get_all_legal_moves(state)
        action_probs = {}
        for m in legal_moves:
            idx = encode_action(state, m)
            action_probs[idx] = float(policy_probs[idx])

        total = sum(action_probs.values())
        if total > 0:
            for a in action_probs:
                action_probs[a] /= total
        elif action_probs:
            n = len(action_probs)
            for a in action_probs:
                action_probs[a] = 1.0 / n

        node.expand(action_probs)
        return value
