"""
play_online.py — Multi-game parallel online play with full move logging,
                 win/loss announcements, and a final session summary.

Architecture:
  - Interactive prompts at startup ask for total games and parallelism.
  - One worker `play_game_worker()` handles a single game end-to-end.
  - Workers run in a ThreadPoolExecutor pool (true sliding window):
      if Game 1 finishes while Game 2 is still running, Game 3 starts immediately.
  - Each move is logged live: [Game X] Turn N: 🤖 Bot moves pawn → e5
  - On game-over a WIN / LOSS banner is printed immediately.
  - When every game is done a session report is printed showing:
      games played · won · lost · per-game improvement · average improvement.

Improvement definition (per-game delta):
  - Each completed game contributes 1 (bot won) or 0 (bot lost).
  - Improvement for game N = result[N] − result[N−1].
  - Game 1: labelled "First game — no prior baseline."
  - Average improvement = mean of deltas for games 2 … N.
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

# ─── Globals shared across threads ────────────────────────────────────────────

_print_lock   = threading.Lock()

# Ordered results: game_id → {'won': bool, 'turns': int, 'moves': list}
# Populated as futures complete; keyed by game_id so we can sort at the end.
_results      = {}
_results_lock = threading.Lock()

# Running tally exposed to the live logging so each game knows current session stats
_session_wins  = 0
_session_games = 0
_tally_lock    = threading.Lock()


# ─── Display helpers ──────────────────────────────────────────────────────────

W = 72  # display width

def log(game_id: int, msg: str):
    """Thread-safe print with [Game X] prefix."""
    with _print_lock:
        print(f"[Game {game_id}] {msg}", flush=True)


def describe_move(move, player_label: str) -> str:
    """Human-readable move description using board-notation coordinates."""
    if isinstance(move, PawnMove):
        col = chr(ord('a') + move.to_col)
        row = move.to_row + 1
        return f"{player_label} moves pawn   →  {col}{row}"
    elif isinstance(move, WallMove):
        col = chr(ord('a') + move.col)
        row = move.row + 1
        orient = "Horizontal" if move.orientation == 'h' else "Vertical  "
        return f"{player_label} places wall  →  {orient} @ {col}{row}"
    else:
        return f"{player_label} plays: {move}"


def _banner(lines: list[str], char: str = "═"):
    """Print a bordered banner block (thread-safe caller responsible for lock)."""
    print(char * W)
    for line in lines:
        print(f"  {line}")
    print(char * W)
    print()


def print_win_loss_banner(game_id: int, won: bool, turns: int,
                          session_wins: int, session_games: int):
    """Immediately print a WIN or LOSS result banner when a game ends."""
    emoji  = "🏆  BOT WON!" if won else "💀  BOT LOST."
    colour = "WIN " if won else "LOSS"
    with _print_lock:
        print()
        print("╔" + "═" * (W - 2) + "╗")
        print(f"║  {f'[Game {game_id}]  {emoji}':^{W-4}}  ║")
        print(f"║  {'Result: ' + colour:^{W-4}}  ║")
        print(f"║  {f'Turns played: {turns}':^{W-4}}  ║")
        print(f"║  {f'Session so far: {session_wins} win(s) / {session_games} game(s)':^{W-4}}  ║")
        print("╚" + "═" * (W - 2) + "╝")
        print()


def print_game_move_history(game_id: int, move_history: list, total_games: int):
    """Print the full move log for one finished game."""
    with _print_lock:
        print()
        header = f"  GAME {game_id} / {total_games} — FULL MOVE HISTORY  "
        print("═" * W)
        print(header.center(W))
        print("═" * W)
        for turn_num, desc in move_history:
            print(f"  Turn {turn_num:>3}: {desc}")
        print("─" * W)
        print()


def print_session_report(results: dict, total_games: int):
    """
    Print the final session summary table with per-game improvement
    and average improvement across all games.
    """
    # Sort by game_id so improvements are computed in order
    ordered = sorted(results.items())   # [(gid, info), ...]

    wins  = sum(1 for _, info in ordered if info['won'])
    losses = total_games - wins

    with _print_lock:
        print()
        print("╔" + "═" * (W - 2) + "╗")
        print(f"║{'  SESSION REPORT  ':^{W}}║")
        print("╠" + "═" * (W - 2) + "╣")

        col_hdr = f"  {'Game':<6}  {'Result':<8}  {'Turns':<6}  {'Improvement':<20}"
        print(f"║{col_hdr:<{W}}║")
        print("╠" + "─" * (W - 2) + "╣")

        improvements = []
        prev_result  = None

        for gid, info in ordered:
            result_int = 1 if info['won'] else 0
            result_str = "✅ WIN " if info['won'] else "❌ LOSS"

            if prev_result is None:
                imp_str = "—  (first game)"
            else:
                delta = result_int - prev_result
                if delta > 0:
                    imp_str = f"+{delta}  ▲ improved"
                elif delta < 0:
                    imp_str = f"{delta}  ▼ declined"
                else:
                    imp_str = f" 0  ─ same"
                improvements.append(delta)

            row = f"  {gid:<6}  {result_str:<8}  {info['turns']:<6}  {imp_str:<20}"
            print(f"║{row:<{W}}║")

            prev_result = result_int

        print("╠" + "═" * (W - 2) + "╣")

        avg_imp = (sum(improvements) / len(improvements)) if improvements else None
        avg_str = f"{avg_imp:+.2f}" if avg_imp is not None else "N/A (only 1 game)"

        summary_lines = [
            f"Games played  : {total_games}",
            f"Games won     : {wins}",
            f"Games lost    : {losses}",
            f"Win rate      : {wins/total_games*100:.1f}%",
            f"Avg improvement (game-to-game delta): {avg_str}",
        ]
        for line in summary_lines:
            print(f"║  {line:<{W-4}}  ║")

        print("╚" + "═" * (W - 2) + "╝")
        print()


# ─── Single-Game Worker ───────────────────────────────────────────────────────

def play_game_worker(game_id: int, settings: dict, mcts_bot, mcts_sims: int,
                     total_games: int):
    """
    Run one complete game end-to-end.
    Returns dict: {'won': bool, 'turns': int, 'moves': list}
    """
    global _session_wins, _session_games

    from adapters.example_site_adapter import ExampleSiteAdapter

    log(game_id, "Starting …")
    move_history = []

    adapter = ExampleSiteAdapter(settings)
    won       = False
    turn_number = 0

    try:
        adapter.connect()
        local_game = QuoridorGame()

        while not adapter.is_game_over() and not local_game.is_terminal:
            turn_number += 1
            board          = local_game.board
            current_player = board.current_player   # 0 = Bot, 1 = Opponent

            is_bot_turn  = (current_player == 0)
            player_label = "🤖 Bot     " if is_bot_turn else "👤 Opponent"

            # Board state (fallback to local engine if adapter returns None)
            state = adapter.get_board_state()
            if state is None:
                state = board

            legal_moves = get_all_legal_moves(state)
            if not legal_moves:
                log(game_id, f"Turn {turn_number}: No legal moves available — ending game.")
                break

            # Select move
            if is_bot_turn:
                if mcts_bot is not None:
                    action_probs = mcts_bot.search(state)
                    best_action  = max(action_probs.items(), key=lambda x: x[1])[0]
                    move = decode_action(state, best_action)
                else:
                    move = random.choice(legal_moves)
            else:
                # Opponent: random until real adapter provides opponent moves
                move = random.choice(legal_moves)

            description = describe_move(move, player_label)
            log(game_id, f"Turn {turn_number:>3}: {description}")
            move_history.append((turn_number, description))

            adapter.make_move(move)
            local_game.apply_move(move)

            time.sleep(0.05)

        # ── Determine result ────────────────────────────────────────────────
        won = (local_game.winner == 0)   # Player 0 = Bot

        with _tally_lock:
            _session_games += 1
            if won:
                _session_wins += 1
            snap_wins  = _session_wins
            snap_games = _session_games

        # ── Immediate win/loss banner ────────────────────────────────────────
        print_win_loss_banner(game_id, won, turn_number, snap_wins, snap_games)

    except Exception as e:
        log(game_id, f"Error: {e}")
        raise
    finally:
        adapter.close()

    return {'won': won, 'turns': turn_number, 'moves': move_history}


# ─── Interactive prompt helper ────────────────────────────────────────────────

def prompt_int(question: str, default: int, minimum: int = 1) -> int:
    """Ask the user for an integer with a default fallback."""
    while True:
        raw = input(f"{question} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val < minimum:
                print(f"  ⚠️  Please enter a number ≥ {minimum}.")
                continue
            return val
        except ValueError:
            print("  ⚠️  Please enter a valid integer.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play Quoridor Online (Parallel)")
    parser.add_argument("--config", type=str, default="configs/settings.yaml",
                        help="Path to settings file")
    parser.add_argument("--games", type=int, default=None,
                        help="Total games to play (skips interactive prompt)")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Max simultaneous games (skips interactive prompt)")
    args = parser.parse_args()

    # ── Load settings ────────────────────────────────────────────────────────
    with open(args.config, 'r') as f:
        settings = yaml.safe_load(f)
    print(f"\nLoaded settings from {args.config}")

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

    # ── Interactive setup (skippable via CLI flags) ──────────────────────────
    print()
    print("─" * W)
    print("  🎮  Quoridor Bot — Session Configuration")
    print("─" * W)

    default_games    = settings['online_play'].get('games_to_play', 4)
    default_parallel = settings['online_play'].get('max_parallel_games', 2)

    if args.games is not None:
        games_to_play = args.games
        print(f"  Total games    : {games_to_play}  (from --games flag)")
    else:
        games_to_play = prompt_int(
            "  How many total games should be played?", default_games)

    if args.parallel is not None:
        max_parallel = args.parallel
        print(f"  Simultaneous   : {max_parallel}  (from --parallel flag)")
    else:
        max_parallel = prompt_int(
            "  How many games should run simultaneously?", default_parallel)

    max_parallel = min(max_parallel, games_to_play)   # cap to total

    print()
    print(f"  ► Running {games_to_play} game(s)  |  {max_parallel} at a time")
    print(f"  ► Pool behaviour: as soon as one game ends, the next queued game starts.")
    print("─" * W)
    print()

    # ── Run games — true sliding-window pool ─────────────────────────────────
    # ThreadPoolExecutor with max_workers=max_parallel + all futures submitted
    # up-front is exactly a sliding-window queue: the pool starts a new task
    # the instant any running task finishes.

    all_results = {}   # gid → {'won', 'turns', 'moves'}
    completed   = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_gid = {
            executor.submit(
                play_game_worker, gid, settings, mcts_bot, mcts_sims, games_to_play
            ): gid
            for gid in range(1, games_to_play + 1)
        }

        for future in concurrent.futures.as_completed(future_to_gid):
            gid = future_to_gid[future]
            completed += 1
            try:
                info = future.result()
                all_results[gid] = info

                # Print the full move log for this finished game
                print_game_move_history(gid, info['moves'], games_to_play)

                remaining_running = len(future_to_gid) - completed
                with _print_lock:
                    if remaining_running > 0:
                        print(f"  ↻  Game {gid} done. "
                              f"{remaining_running} game(s) still running / queued.\n")

            except Exception as exc:
                with _print_lock:
                    print(f"\n[Game {gid}] ❌ Failed: {exc}\n")
                all_results[gid] = {'won': False, 'turns': 0, 'moves': []}

    # ── Session report ───────────────────────────────────────────────────────
    print_session_report(all_results, games_to_play)

    # ── Checkpointing (placeholder) ──────────────────────────────────────────
    save_interval       = settings.get('models', {}).get('save_interval', 100)
    super_save_interval = settings.get('models', {}).get('super_save_interval', 1000)
    if games_to_play % save_interval == 0:
        print(f"Reached {save_interval} games — saving checkpoint …")
    if games_to_play % super_save_interval == 0:
        print(f"Reached {super_save_interval} games — archiving super-checkpoint …")

    print("Session complete. Goodbye.\n")


if __name__ == "__main__":
    main()
