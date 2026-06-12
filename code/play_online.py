"""
play_online.py — Multi-game parallel online play with full move logging.

Architecture:
  - One worker function `play_game_worker()` handles a single game end-to-end.
  - Multiple workers run concurrently via ThreadPoolExecutor.
  - Each move is printed live with a [Game X] prefix.
  - A complete, clean move history is printed at the end of each game.
  - The fallback path (when the adapter cannot read the real board) simulates
    an alternating game locally, with "Bot" and "Opponent" labelled correctly.
"""

import argparse
import yaml
import time
import random
import threading
import concurrent.futures
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from search.mcts import MCTS
from engine.game import QuoridorGame
from engine.moves import PawnMove, WallMove
from engine.rules import get_all_legal_moves
from models.board_encoder import decode_action
from training.model_registry import load_best_model
from utils.model_manager import ModelManager

# ─── Helpers ─────────────────────────────────────────────────────────────────

_print_lock = threading.Lock()

def log(game_id: int, msg: str):
    """Thread-safe print with game prefix."""
    with _print_lock:
        print(f"[Game {game_id}] {msg}", flush=True)


def describe_move(move, player_label: str) -> str:
    """Return a human-readable description of a move."""
    if isinstance(move, PawnMove):
        col_letter = chr(ord('a') + move.to_col)   # column a–i
        row_number  = move.to_row + 1               # row 1–9
        return f"{player_label} moves pawn  →  {col_letter}{row_number}"
    elif isinstance(move, WallMove):
        col_letter  = chr(ord('a') + move.to_col if hasattr(move, 'to_col') else ord('a') + move.col)
        row_number  = (move.to_row if hasattr(move, 'to_row') else move.row) + 1
        orientation = "Horizontal" if move.orientation == 'h' else "Vertical"
        return (f"{player_label} places wall  →  "
                f"{orientation} @ {col_letter}{row_number}")
    else:
        return f"{player_label} plays: {move}"


# ─── Single-Game Worker ───────────────────────────────────────────────────────

def play_game_worker(game_id: int, settings: dict, mcts_bot, mcts_sims: int):
    """
    Run one complete game.
    Returns a list of move description strings (the game's full history).
    """
    from adapters.example_site_adapter import ExampleSiteAdapter

    log(game_id, "Starting …")

    move_history = []   # (turn_number, description)

    adapter = ExampleSiteAdapter(settings)
    try:
        adapter.connect()
        local_game  = QuoridorGame()
        turn_number = 0

        while not adapter.is_game_over() and not local_game.is_terminal:
            turn_number += 1
            board = local_game.board
            current_player = board.current_player   # 0 = Bot, 1 = Opponent

            # ── Determine whose turn it is ──────────────────────────────────
            # adapter.is_my_turn() returns True  → it's the Bot's turn.
            # In fallback mode the adapter always returns True, so we use the
            # local game's current_player to simulate alternation properly.
            adapter_says_my_turn = adapter.is_my_turn()

            # Respect the actual local state; override adapter for simulation
            is_bot_turn  = (current_player == 0)
            player_label = "🤖 Bot     " if is_bot_turn else "👤 Opponent"

            # ── Get board state ─────────────────────────────────────────────
            state = adapter.get_board_state()
            if state is None:
                state = board   # fallback: use local engine state

            # ── Compute move ────────────────────────────────────────────────
            legal_moves = get_all_legal_moves(state)
            if not legal_moves:
                log(game_id, f"Turn {turn_number}: No legal moves — game over.")
                break

            if is_bot_turn:
                if mcts_bot is not None:
                    action_probs = mcts_bot.search(state)
                    best_action  = max(action_probs.items(), key=lambda x: x[1])[0]
                    move = decode_action(state, best_action)
                else:
                    move = random.choice(legal_moves)
            else:
                # Opponent: random in simulation (until adapter exposes real moves)
                move = random.choice(legal_moves)

            # ── Apply and log ───────────────────────────────────────────────
            description = describe_move(move, player_label)
            log(game_id, f"Turn {turn_number:>3}: {description}")
            move_history.append((turn_number, description))

            adapter.make_move(move)
            local_game.apply_move(move)

            time.sleep(0.05)   # small yield so threads interleave naturally

        # ── Game-over summary ───────────────────────────────────────────────
        winner_label = "🤖 Bot" if local_game.winner == 0 else "👤 Opponent"
        if local_game.winner is None:
            winner_label = "Unknown"

        log(game_id, f"Finished after {turn_number} turns. Winner: {winner_label}")

    except Exception as e:
        log(game_id, f"Error: {e}")
        raise
    finally:
        adapter.close()

    return move_history


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_game_summary(game_id: int, move_history: list, max_games: int):
    """Print a formatted move history for one finished game."""
    width = 70
    header = f"  GAME {game_id} / {max_games} — FULL MOVE HISTORY  "
    with _print_lock:
        print()
        print("═" * width)
        print(header.center(width))
        print("═" * width)
        for turn_num, desc in move_history:
            print(f"  Turn {turn_num:>3}: {desc}")
        print("─" * width)
        print()


def main():
    parser = argparse.ArgumentParser(description="Play Quoridor Online (Parallel)")
    parser.add_argument("--config", type=str, default="configs/settings.yaml",
                        help="Path to settings file")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Override max parallel games (default: from settings)")
    args = parser.parse_args()

    # ── Load settings ────────────────────────────────────────────────────────
    with open(args.config, 'r') as f:
        settings = yaml.safe_load(f)
    print(f"Loaded settings from {args.config}")

    # ── Model management ─────────────────────────────────────────────────────
    model_manager = ModelManager(args.config)
    model_manager.cleanup_old_models()
    print(f"Active model path: {model_manager.get_active_model_path()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    champion_result = load_best_model(device=device)
    if champion_result is not None:
        model, champion_meta = champion_result
        model.eval()
        version = champion_meta.get('version', '?')
        elo     = champion_meta.get('elo', '?')
        print(f"✅ AlphaZero champion loaded — version={version}, elo={elo}")
        mcts_sims = settings.get('mcts', {}).get('simulations', 100)
        mcts_bot  = MCTS(model, num_simulations=mcts_sims)
    else:
        print("⚠️  No trained AlphaZero model found — falling back to random moves.")
        mcts_bot  = None
        mcts_sims = 0

    # ── Concurrency config ───────────────────────────────────────────────────
    games_to_play  = settings['online_play'].get('games_to_play', 1)
    max_parallel   = args.parallel or settings['online_play'].get('max_parallel_games', 2)
    max_parallel   = min(max_parallel, games_to_play)   # never more than needed

    print(f"\n🎮 Starting {games_to_play} game(s) with up to {max_parallel} in parallel …\n")
    print("=" * 70)

    # ── Run games in parallel ────────────────────────────────────────────────
    results = {}   # game_id → move_history list

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_gid = {
            executor.submit(play_game_worker, gid, settings, mcts_bot, mcts_sims): gid
            for gid in range(1, games_to_play + 1)
        }

        for future in concurrent.futures.as_completed(future_to_gid):
            gid = future_to_gid[future]
            try:
                move_history = future.result()
                results[gid] = move_history
                print_game_summary(gid, move_history, games_to_play)
            except Exception as exc:
                with _print_lock:
                    print(f"\n[Game {gid}] ❌ Failed with exception: {exc}\n")

    # ── Checkpointing (placeholder) ──────────────────────────────────────────
    total_games = games_to_play
    save_interval       = settings.get('models', {}).get('save_interval', 100)
    super_save_interval = settings.get('models', {}).get('super_save_interval', 1000)

    if total_games % save_interval == 0:
        print(f"Reached {save_interval} games — saving checkpoint …")
    if total_games % super_save_interval == 0:
        print(f"Reached {super_save_interval} games — archiving super-checkpoint …")

    print("All games finished. Goodbye.")


if __name__ == "__main__":
    main()
