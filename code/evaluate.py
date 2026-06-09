"""
Evaluation entrypoint for Final SINN.

Supports:
  - Evaluating arbitrary checkpoints
  - Evaluating best_model.pt via --model_a best or --model_b best
  - Comparing two checkpoints head-to-head
  - minimax keyword for Minimax opponent
  - Persistent MCTS instances for TT reuse
  - Updates Elo in metadata.json
"""

import argparse
import os
import sys
import math
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import random

from utils.logger import setup_logger
from evaluation.elo import EloSystem
from training.model_registry import (
    load_checkpoint,
    load_metadata,
    save_metadata,
    BEST_MODEL_PATH,
)


def load_config(config_path):
    if not os.path.exists(config_path):
        return {
            'logging': {'level': 'INFO', 'log_dir': 'data/logs',
                        'experiments_dir': 'data/experiments'},
            'mcts': {'simulations': 50},
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_agent(model_path, device, mcts_sims):
    """
    Load an agent. Returns ('mcts', MCTS_instance) or ('minimax', None).
    Handles 'best', 'minimax', or a file path.
    """
    from models.quoridor_net import QuoridorNet
    from search.mcts import MCTS

    if model_path.lower() == 'minimax':
        return 'minimax', None

    actual_path = model_path
    if model_path.lower() == 'best':
        if not os.path.exists(BEST_MODEL_PATH):
            print(f"Error: No champion model found at {BEST_MODEL_PATH}")
            sys.exit(1)
        actual_path = BEST_MODEL_PATH

    model = QuoridorNet().to(device)
    if os.path.exists(actual_path):
        ckpt = load_checkpoint(actual_path, device=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        logging.getLogger("QuoridorAI").warning(
            f"Weights {actual_path} not found. Using random model!")
    model.eval()
    # Persistent MCTS for TT reuse across the entire evaluation
    mcts = MCTS(model, num_simulations=mcts_sims)
    return 'mcts', mcts


def get_move(agent_type, agent, board, player):
    if agent_type == 'minimax':
        from search.minimax import alphabeta
        _, best_move = alphabeta(
            board, depth=4, alpha=-math.inf, beta=math.inf,
            maximizing_player=True, root_player=player)
        return best_move
    else:
        from models.board_encoder import decode_action
        action_probs = agent.search(board)
        best_idx = max(action_probs.items(), key=lambda x: x[1])[0]
        return decode_action(board, best_idx)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quoridor AI Models")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model_a', type=str, required=True,
                        help='Path to model A, or "best" or "minimax"')
    parser.add_argument('--model_b', type=str, required=True,
                        help='Path to model B, or "best" or "minimax"')
    parser.add_argument('--games', type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    logger_obj = setup_logger(config)
    logger = logger_obj.logger
    elo = EloSystem()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mcts_sims = config.get('mcts', {}).get('simulations', 50)

    logger.info(f"Evaluation: {args.model_a} vs {args.model_b} over {args.games} games")

    type_a, agent_a = load_agent(args.model_a, device, mcts_sims)
    type_b, agent_b = load_agent(args.model_b, device, mcts_sims)

    from engine.game import QuoridorGame

    score_a = 0
    score_b = 0

    for g in range(args.games):
        game = QuoridorGame()
        pA = 0 if g % 2 == 0 else 1
        pB = 1 - pA

        # Reset MCTS TTs each game for clean evaluation
        if type_a == 'mcts':
            agent_a.tt = {}
        if type_b == 'mcts':
            agent_b.tt = {}

        while not game.is_terminal:
            curr_p = game.board.current_player
            if curr_p == pA:
                move = get_move(type_a, agent_a, game.board, pA)
            else:
                move = get_move(type_b, agent_b, game.board, pB)

            if move is None:
                legal = game.get_legal_moves()
                move = random.choice(legal)

            game.apply_move(move)

        winner = game.winner
        if winner == pA:
            sa = 1
            score_a += 1
            logger.info(f"Game {g+1}: Model A wins as P{pA}")
        else:
            sa = 0
            score_b += 1
            logger.info(f"Game {g+1}: Model B wins as P{pB}")

        elo.update_ratings(args.model_a, args.model_b, sa)

    elo_a = elo.get_rating(args.model_a)
    elo_b = elo.get_rating(args.model_b)

    logger.info(f"Final Score: A={score_a} | B={score_b}")
    logger.info(f"Win Rate A: {score_a/max(args.games,1):.1%}")
    logger.info(f"Elo: A={elo_a:.1f} | B={elo_b:.1f}")

    # Update metadata with Elo if 'best' was used
    meta = load_metadata()
    if args.model_a.lower() == 'best':
        meta['champion_elo'] = elo_a
    if args.model_b.lower() == 'best':
        meta['champion_elo'] = elo_b
    save_metadata(meta)


if __name__ == "__main__":
    main()
