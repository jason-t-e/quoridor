"""
Training entrypoint for Final SINN.

Behaviour:
  - ALWAYS resumes from best_model.pt if it exists (unless --from-scratch).
  - Periodic checkpointing every `checkpoint_interval` games.
  - Auto-promotes the current model to champion if it wins >55% of eval games.
  - Graceful shutdown on Ctrl+C — saves current checkpoint before exiting.
  - NEVER overwrites champion accidentally.
"""

import argparse
import os
import sys
import signal
import yaml
import torch
import torch.optim as optim
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from models.quoridor_net import QuoridorNet
from training.model_registry import (
    load_or_init_for_training,
    save_checkpoint,
    promote_to_champion,
    load_metadata,
    save_metadata,
    load_best_model,
    next_version,
    migrate_legacy_checkpoints,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
)
from training.experience_buffer import ExperienceBuffer
from training.self_play_worker import play_self_play_game
from training.trainer import train_step

logger = logging.getLogger("QuoridorAI")


def load_config(config_path):
    if not os.path.exists(config_path):
        return {
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'checkpoint_dir': 'data/checkpoints',
                'games_per_epoch': 10,
                'epochs': 10,
                'checkpoint_interval': 50,
                'eval_interval': 100,
                'eval_games': 40,
                'promotion_threshold': 0.55,
                'resume': True,
                'auto_promote': True,
            },
            'mcts': {'simulations': 50},
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_models(current_model, champion_model, num_games, mcts_sims, device):
    """
    Play num_games between current_model and champion_model.
    Returns current_model win rate.
    """
    from search.mcts import MCTS
    from engine.game import QuoridorGame
    from models.board_encoder import decode_action

    current_model.eval()
    champion_model.eval()

    current_wins = 0

    for g in range(num_games):
        game = QuoridorGame()
        # Alternate sides
        current_is_p0 = (g % 2 == 0)

        mcts_current = MCTS(current_model, num_simulations=mcts_sims)
        mcts_champion = MCTS(champion_model, num_simulations=mcts_sims)

        while not game.is_terminal:
            p = game.board.current_player
            is_current = (p == 0 and current_is_p0) or (p == 1 and not current_is_p0)
            mcts_agent = mcts_current if is_current else mcts_champion

            action_probs = mcts_agent.search(game.board)
            best_action = max(action_probs.items(), key=lambda x: x[1])[0]
            move = decode_action(game.board, best_action)
            game.apply_move(move)

        winner = game.winner
        current_won = (winner == 0 and current_is_p0) or (winner == 1 and not current_is_p0)
        if current_won:
            current_wins += 1

    return current_wins / max(num_games, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Final SINN Quoridor AI")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--games-per-epoch', type=int, default=None)
    parser.add_argument('--from-scratch', action='store_true',
                        help='Ignore existing checkpoints and start fresh')
    parser.add_argument('--no-promote', action='store_true',
                        help='Disable auto-promotion of improved models')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger(config)

    tcfg = config.get('training', {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Migrate any legacy checkpoints before anything else
    migrate_legacy_checkpoints()

    # Load or init model
    lr = tcfg.get('learning_rate', 0.001)
    model, optimizer, meta = load_or_init_for_training(
        device=device, lr=lr, from_scratch=args.from_scratch)

    # Training parameters
    epochs = args.epochs or tcfg.get('epochs', 10)
    games_per_epoch = args.games_per_epoch or tcfg.get('games_per_epoch', 10)
    batch_size = tcfg.get('batch_size', 32)
    mcts_sims = config.get('mcts', {}).get('simulations', 50)
    ckpt_interval = tcfg.get('checkpoint_interval', 50)
    eval_interval = tcfg.get('eval_interval', 100)
    eval_games = tcfg.get('eval_games', 40)
    promotion_threshold = tcfg.get('promotion_threshold', 0.55)
    auto_promote = not args.no_promote and tcfg.get('auto_promote', True)

    buffer = ExperienceBuffer(capacity=100000)

    training_step = meta.get('training_step', 0)
    games_played = meta.get('games_played', 0)
    current_version = meta.get('version', 'v1')
    current_elo = meta.get('elo', 1200.0)

    # Graceful shutdown handler
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        logger.info("\nShutdown requested — saving checkpoint...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"Starting training: {epochs} epochs × {games_per_epoch} games/epoch")
    logger.info(f"Resuming from v{current_version}, step={training_step}, games={games_played}, elo={current_elo:.1f}")

    total_games_this_session = 0

    for epoch in range(1, epochs + 1):
        if shutdown_requested:
            break

        logger.info(f"--- Epoch {epoch}/{epochs} ---")

        for g in range(1, games_per_epoch + 1):
            if shutdown_requested:
                break

            logger.info(f"Self-play game {g}/{games_per_epoch} (total: {games_played + 1})...")
            game_data = play_self_play_game(model, mcts_simulations=mcts_sims)
            for state, policy, value in game_data:
                buffer.add(state, policy, value)

            games_played += 1
            total_games_this_session += 1

            # Train on batch
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                p_loss, v_loss = train_step(model, optimizer, batch, device=str(device))
                training_step += 1
                if training_step % 10 == 0:
                    logger.info(f"  Step {training_step}: policy_loss={p_loss:.4f}, value_loss={v_loss:.4f}")

            # Periodic checkpoint
            if total_games_this_session % ckpt_interval == 0:
                current_version = next_version(current_version)
                ckpt_path = save_checkpoint(
                    model, optimizer,
                    training_step=training_step,
                    games_played=games_played,
                    elo=current_elo,
                    version=current_version,
                    config=tcfg,
                )
                logger.info(f"Periodic checkpoint saved: {ckpt_path}")

            # Evaluation & promotion
            if auto_promote and total_games_this_session % eval_interval == 0:
                logger.info(f"Running evaluation ({eval_games} games)...")

                champion_result = load_best_model(device=device)
                if champion_result is not None:
                    champion_model, champion_meta = champion_result
                    win_rate = evaluate_models(
                        model, champion_model, eval_games, mcts_sims, device)
                    logger.info(f"Current vs Champion: {win_rate:.1%} win rate")

                    if win_rate > promotion_threshold:
                        current_version = next_version(current_version)
                        # Simple Elo update
                        current_elo += (win_rate - 0.5) * 100
                        ckpt_path = save_checkpoint(
                            model, optimizer,
                            training_step=training_step,
                            games_played=games_played,
                            elo=current_elo,
                            version=current_version,
                            config=tcfg,
                        )
                        promote_to_champion(ckpt_path)
                        logger.info(f"★ New champion promoted: v{current_version} (elo={current_elo:.1f})")
                    else:
                        logger.info("Current model did not beat champion. Continuing training.")
                else:
                    # No champion yet — promote current
                    current_version = next_version(current_version)
                    ckpt_path = save_checkpoint(
                        model, optimizer,
                        training_step=training_step,
                        games_played=games_played,
                        elo=current_elo,
                        version=current_version,
                        config=tcfg,
                    )
                    promote_to_champion(ckpt_path)
                    logger.info(f"★ First champion established: v{current_version}")

    # Final save
    current_version = next_version(current_version)
    final_path = save_checkpoint(
        model, optimizer,
        training_step=training_step,
        games_played=games_played,
        elo=current_elo,
        version=current_version,
        config=tcfg,
    )
    logger.info(f"Final checkpoint saved: {final_path}")

    # If no champion exists yet, promote final checkpoint
    if not os.path.exists(BEST_MODEL_PATH):
        promote_to_champion(final_path)
        logger.info(f"★ First champion established: v{current_version}")
    elif auto_promote:
        # One final evaluation
        champion_result = load_best_model(device=device)
        if champion_result is not None:
            champion_model, _ = champion_result
            win_rate = evaluate_models(
                model, champion_model, eval_games, mcts_sims, device)
            logger.info(f"Final evaluation: {win_rate:.1%} win rate vs champion")
            if win_rate > promotion_threshold:
                current_elo += (win_rate - 0.5) * 100
                final_path = save_checkpoint(
                    model, optimizer,
                    training_step=training_step,
                    games_played=games_played,
                    elo=current_elo,
                    version=current_version,
                    config=tcfg,
                    path=final_path,
                )
                promote_to_champion(final_path)
                logger.info(f"★ New champion after final eval: v{current_version} (elo={current_elo:.1f})")

    # Update global metadata
    global_meta = load_metadata()
    global_meta['total_training_steps'] = training_step
    global_meta['total_games_played'] = games_played
    save_metadata(global_meta)

    logger.info(f"Training complete. Total steps: {training_step}, total games: {games_played}")


if __name__ == "__main__":
    main()
