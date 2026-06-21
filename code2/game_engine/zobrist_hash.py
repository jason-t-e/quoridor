import numpy as np
from .moves import PawnMove, WallMove

class ZobristHasher:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed=seed)
        self.pawn_table = rng.integers(1, 2**64-1, size=(2, 9, 9), dtype=np.uint64)
        self.hwall_table = rng.integers(1, 2**64-1, size=(8, 8), dtype=np.uint64)
        self.vwall_table = rng.integers(1, 2**64-1, size=(8, 8), dtype=np.uint64)
        self.walls_table = rng.integers(1, 2**64-1, size=(2, 11), dtype=np.uint64)
        self.player_table = rng.integers(1, 2**64-1, size=(2,), dtype=np.uint64)

    def full_hash(self, board) -> int:
        """Called once per game at initialization."""
        h = np.uint64(0)
        for p in [0, 1]:
            r, c = board.pawn_positions[p]
            h ^= self.pawn_table[p, r, c]
            h ^= self.walls_table[p, board.walls_remaining[p]]
        for (r, c) in board.h_walls:
            h ^= self.hwall_table[r, c]
        for (r, c) in board.v_walls:
            h ^= self.vwall_table[r, c]
        h ^= self.player_table[board.current_player]
        return int(h)

    def incremental_update(self, current_hash: int, board_before, move) -> int:
        """O(1). Called inside apply_move() automatically."""
        h = np.uint64(current_hash)
        player = board_before.current_player
        
        if isinstance(move, PawnMove):
            r_old, c_old = board_before.pawn_positions[player]
            h ^= self.pawn_table[player, r_old, c_old]
            h ^= self.pawn_table[player, move.to_row, move.to_col]
        elif isinstance(move, WallMove):
            w_old = board_before.walls_remaining[player]
            h ^= self.walls_table[player, w_old]
            h ^= self.walls_table[player, w_old - 1]
            table = self.hwall_table if move.orientation == 'h' else self.vwall_table
            h ^= table[move.row, move.col]
            
        h ^= self.player_table[board_before.current_player]
        h ^= self.player_table[1 - board_before.current_player]
        return int(h)

# Global singleton
HASHER = ZobristHasher()
