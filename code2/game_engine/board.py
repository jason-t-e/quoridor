import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple
from .moves import Move, PawnMove, WallMove
from .zobrist_hash import HASHER

@dataclass
class BoardState:
    pawn_positions: Dict[int, Tuple[int, int]] = field(default_factory=lambda: {0: (0, 4), 1: (8, 4)})
    h_walls: Set[Tuple[int, int]] = field(default_factory=set)
    v_walls: Set[Tuple[int, int]] = field(default_factory=set)
    walls_remaining: Dict[int, int] = field(default_factory=lambda: {0: 10, 1: 10})
    current_player: int = 0
    move_history: List[Move] = field(default_factory=list)
    turn: int = 0
    current_hash: int = 0

    def __post_init__(self):
        if self.current_hash == 0:
            self.current_hash = HASHER.full_hash(self)

    def apply_move(self, move: Move) -> 'BoardState':
        """Returns a new BoardState with the move applied."""
        new_hash = HASHER.incremental_update(self.current_hash, self, move)
        
        new_pawn_positions = dict(self.pawn_positions)
        new_h_walls = set(self.h_walls)
        new_v_walls = set(self.v_walls)
        new_walls_remaining = dict(self.walls_remaining)
        
        if isinstance(move, PawnMove):
            new_pawn_positions[move.player] = (move.to_row, move.to_col)
        elif isinstance(move, WallMove):
            if move.orientation == 'h':
                new_h_walls.add((move.row, move.col))
            else:
                new_v_walls.add((move.row, move.col))
            new_walls_remaining[move.player] -= 1

        new_history = list(self.move_history)
        new_history.append(move)

        return BoardState(
            pawn_positions=new_pawn_positions,
            h_walls=new_h_walls,
            v_walls=new_v_walls,
            walls_remaining=new_walls_remaining,
            current_player=1 - self.current_player,
            move_history=new_history,
            turn=self.turn + 1,
            current_hash=new_hash
        )

    def is_terminal(self) -> bool:
        return self.pawn_positions[0][0] == 8 or self.pawn_positions[1][0] == 0

    @property
    def winner(self) -> int | None:
        if self.pawn_positions[0][0] == 8:
            return 0
        if self.pawn_positions[1][0] == 0:
            return 1
        return None

def board_to_tensor(board: BoardState) -> np.ndarray:
    """Converts BoardState to [10 x 9 x 9] tensor from current player's perspective."""
    from .pathfinder import bfs_distance_map
    
    tensor = np.zeros((10, 9, 9), dtype=np.float32)
    p = board.current_player
    opp = 1 - p
    
    # 0: Own pawn
    tensor[0, board.pawn_positions[p][0], board.pawn_positions[p][1]] = 1.0
    # 1: Opp pawn
    tensor[1, board.pawn_positions[opp][0], board.pawn_positions[opp][1]] = 1.0
    
    # Walls
    for r, c in board.h_walls:
        tensor[2, r, c] = 1.0
        tensor[2, r, c+1] = 1.0
    for r, c in board.v_walls:
        tensor[3, r, c] = 1.0
        tensor[3, r+1, c] = 1.0
        
    # BFS maps
    dist_p = bfs_distance_map(board, p)
    dist_opp = bfs_distance_map(board, opp)
    
    max_dist = max(np.max(dist_p), np.max(dist_opp), 1)
    tensor[4] = dist_p / max_dist
    tensor[5] = dist_opp / max_dist
    
    # Scalars broadcasted
    tensor[6, :, :] = board.walls_remaining[p] / 10.0
    tensor[7, :, :] = board.walls_remaining[opp] / 10.0
    
    # Phase
    phase = 0.0
    if board.turn > 10: phase = 0.5
    if board.turn > 30: phase = 1.0
    tensor[8, :, :] = phase
    
    # Current player
    tensor[9, :, :] = float(p)
    
    # Rotate if player 1 so goal is always at row 8?
    # TRD says "flipped so 'my goal' is always row 8"
    if p == 1:
        # Flip rows for all channels
        tensor = tensor[:, ::-1, :]
        
    return tensor
