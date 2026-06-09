from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple
from engine.moves import Move

@dataclass
class BoardState:
    # Player 0 starts at (0, 4) and wants to reach row 8
    # Player 1 starts at (8, 4) and wants to reach row 0
    pawn_positions: Dict[int, Tuple[int, int]] = field(default_factory=lambda: {0: (0, 4), 1: (8, 4)})
    h_walls: Set[Tuple[int, int]] = field(default_factory=set)
    v_walls: Set[Tuple[int, int]] = field(default_factory=set)
    walls_remaining: Dict[int, int] = field(default_factory=lambda: {0: 10, 1: 10})
    current_player: int = 0
    move_history: List[Move] = field(default_factory=list)
    turn: int = 0
    current_hash: int = 0
    
    def __post_init__(self):
        # Compute Zobrist hash from scratch when a new BoardState is created
        # (but not when cloning — clone sets current_hash directly)
        if self.current_hash == 0 and self.turn == 0:
            from engine.zobrist import full_hash
            self.current_hash = full_hash(self)
    
    def clone(self) -> 'BoardState':
        return BoardState(
            pawn_positions=self.pawn_positions.copy(),
            h_walls=self.h_walls.copy(),
            v_walls=self.v_walls.copy(),
            walls_remaining=self.walls_remaining.copy(),
            current_player=self.current_player,
            move_history=self.move_history.copy(),
            turn=self.turn,
            current_hash=self.current_hash
        )

