from dataclasses import dataclass
from typing import Callable
from game_engine.board import BoardState
from game_engine.moves import Move, is_wall_move, is_pawn_move
from game_engine.pathfinder import path_differential, shortest_path_to_goal

@dataclass
class StrategyRule:
    name: str
    phase: int # 0=All, 1=Opening, 2=Mid, 3=End
    condition: Callable[[BoardState], bool]
    agreement: Callable[[BoardState, Move], float]
    weight: float = 1.0

# Simplified helper functions
def is_central_column(move: Move) -> bool:
    if is_pawn_move(move):
        return 3 <= move.to_col <= 5
    return False

def shortens_own_path(board: BoardState, move: Move) -> bool:
    p = board.current_player
    dist_before = shortest_path_to_goal(board, p)
    dist_after = shortest_path_to_goal(board.apply_move(move), p)
    return dist_after < dist_before

STRATEGY_RULES = [
    StrategyRule(
        name="path_race_advance",
        phase=0,
        condition=lambda b: path_differential(b) < 0,
        agreement=lambda b, a: (
            1.0 if is_pawn_move(a) and shortens_own_path(b, a) else 0.0
        )
    ),
    StrategyRule(
        name="standard_opening_center",
        phase=1,
        condition=lambda b: b.turn < 6,
        agreement=lambda b, a: (
            1.0 if is_pawn_move(a) and is_central_column(a) else
            0.3 if is_pawn_move(a) else 0.0
        )
    )
]

class StrategyGuide:
    def __init__(self, rules=None):
        self.rules = rules or STRATEGY_RULES
        
    def compute_strategy_vector(self, board: BoardState, counter_rec=None) -> list[float]:
        # TRD specifies a 16-dim strategy vector
        vec = [0.0] * 16
        # Just stubbing it with some data
        vec[5] = path_differential(board) / 9.0
        
        if board.turn <= 10: vec[7] = 1.0
        elif board.turn <= 30: vec[8] = 1.0
        else: vec[9] = 1.0
        
        return vec
