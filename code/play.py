"""
Play entrypoint for Final SINN.

RULES:
  - If best_model.pt exists, load it immediately and start the game.
  - If no champion exists, display a message and exit.
  - NEVER start a training process automatically.
"""

import argparse
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from training.model_registry import (
    load_best_model,
    load_metadata,
    migrate_legacy_checkpoints,
    BEST_MODEL_PATH,
)


def load_config(config_path):
    if not os.path.exists(config_path):
        return {
            'logging': {'level': 'INFO', 'log_dir': 'data/logs',
                        'experiments_dir': 'data/experiments'},
            'mcts': {'simulations': 400},
        }
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Play Quoridor AI (Final SINN)")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=False, default=None,
                        help='Override: path to a specific model checkpoint')
    parser.add_argument('--player', type=int, default=0, choices=[0, 1],
                        help='0 to go first, 1 to go second')
    args = parser.parse_args()

    config = load_config(args.config)
    logger_obj = setup_logger(config)
    logger = logger_obj.logger

    # Migrate legacy checkpoints if needed
    migrate_legacy_checkpoints()

    # Determine model to use
    model_path = args.model

    if model_path is None:
        # Default: use champion
        if not os.path.exists(BEST_MODEL_PATH):
            print("\n" + "=" * 60)
            print("  No trained model found.")
            print("  Please train the system first:")
            print()
            print("    python code/train.py")
            print()
            print("  This will create a champion model for gameplay.")
            print("=" * 60 + "\n")
            sys.exit(1)

        model_path = BEST_MODEL_PATH
        meta = load_metadata()
        logger.info(f"Champion loaded: v{meta.get('champion_version', '?')}, "
                     f"Elo={meta.get('champion_elo', '?')}, "
                     f"Games={meta.get('champion_games_played', '?')}")

    logger.info(f"Starting Quoridor GUI. Model: {model_path}, Human player: {args.player}")

    from engine.game import QuoridorGame
    from ui.pygame_gui import PygameGUI

    game = QuoridorGame()
    gui = PygameGUI(game, default_model_path=model_path,
                    human_player=args.player, config=config)
    gui.run()

    logger.info("Game over.")


if __name__ == "__main__":
    main()
