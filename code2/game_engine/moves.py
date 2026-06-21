from dataclasses import dataclass

@dataclass(frozen=True)
class PawnMove:
    player: int
    to_row: int
    to_col: int

@dataclass(frozen=True)
class WallMove:
    player: int
    row: int
    col: int
    orientation: str  # 'h' or 'v'

Move = PawnMove | WallMove

def is_pawn_move(move: Move) -> bool:
    return isinstance(move, PawnMove)

def is_wall_move(move: Move) -> bool:
    return isinstance(move, WallMove)
