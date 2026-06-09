"""
Zobrist Hashing for Quoridor.

Provides O(1) incremental board state hashing for use in Transposition Tables.
Uses pre-computed random 64-bit integers XORed together to produce a unique hash
for each board configuration.

Components hashed:
  - Pawn positions: 2 players × 81 squares
  - Horizontal walls: 8×8 = 64 positions
  - Vertical walls: 8×8 = 64 positions
  - Walls remaining: 2 players × 11 possible counts (0..10)
  - Side to move: 1 key
"""

import random

# Fixed seed so hashes are deterministic across runs and processes.
_RNG = random.Random(0xDEADBEEF_QUORIDOR)

def _rand64() -> int:
    return _RNG.getrandbits(64)


# Pre-computed tables ---------------------------------------------------

# Pawn positions: PAWN_KEYS[player][row * 9 + col]
PAWN_KEYS = [[_rand64() for _ in range(81)] for _ in range(2)]

# Wall positions: H_WALL_KEYS[row * 8 + col],  V_WALL_KEYS[row * 8 + col]
H_WALL_KEYS = [_rand64() for _ in range(64)]
V_WALL_KEYS = [_rand64() for _ in range(64)]

# Walls remaining: WALLS_REM_KEYS[player][count]  (count 0..10)
WALLS_REM_KEYS = [[_rand64() for _ in range(11)] for _ in range(2)]

# Side to move (XOR this to flip)
SIDE_TO_MOVE_KEY = _rand64()


# Public API -----------------------------------------------------------

def full_hash(board) -> int:
    """Compute the complete Zobrist hash from scratch."""
    h = 0

    # Pawn positions
    for player in (0, 1):
        r, c = board.pawn_positions[player]
        h ^= PAWN_KEYS[player][r * 9 + c]

    # Horizontal walls
    for (wr, wc) in board.h_walls:
        h ^= H_WALL_KEYS[wr * 8 + wc]

    # Vertical walls
    for (wr, wc) in board.v_walls:
        h ^= V_WALL_KEYS[wr * 8 + wc]

    # Walls remaining
    for player in (0, 1):
        h ^= WALLS_REM_KEYS[player][board.walls_remaining[player]]

    # Side to move (hash in only when player 1 is to move)
    if board.current_player == 1:
        h ^= SIDE_TO_MOVE_KEY

    return h


def incremental_pawn_move(old_hash: int, player: int, old_pos: tuple, new_pos: tuple) -> int:
    """Update hash after a pawn move.  XOR out old position, XOR in new."""
    h = old_hash
    h ^= PAWN_KEYS[player][old_pos[0] * 9 + old_pos[1]]
    h ^= PAWN_KEYS[player][new_pos[0] * 9 + new_pos[1]]
    return h


def incremental_wall_place(old_hash: int, player: int, orientation: str, pos: tuple, old_wall_count: int) -> int:
    """Update hash after placing a wall."""
    h = old_hash
    r, c = pos

    # XOR in the wall
    if orientation == 'h':
        h ^= H_WALL_KEYS[r * 8 + c]
    else:
        h ^= V_WALL_KEYS[r * 8 + c]

    # XOR out old wall count, XOR in new
    h ^= WALLS_REM_KEYS[player][old_wall_count]
    h ^= WALLS_REM_KEYS[player][old_wall_count - 1]

    return h


def incremental_turn(old_hash: int) -> int:
    """Toggle the side-to-move bit."""
    return old_hash ^ SIDE_TO_MOVE_KEY
