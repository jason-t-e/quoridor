from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Move:
    pass

@dataclass(frozen=True)
class PawnMove(Move):
    to_row: int
    to_col: int
    
    def __repr__(self):
        return f"Pawn({self.to_row}, {self.to_col})"

@dataclass(frozen=True)
class WallMove(Move):
    row: int
    col: int
    orientation: Literal['h', 'v']
    
    def __repr__(self):
        return f"Wall({self.row}, {self.col}, {self.orientation})"
