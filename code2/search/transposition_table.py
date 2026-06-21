from typing import Dict, Any, Tuple
from enum import Enum

class BoundType(Enum):
    EXACT = 1
    LOWER = 2
    UPPER = 3

class TTEntry:
    def __init__(self, value: float, best_action: Any, bound: BoundType, depth: int):
        self.value = value
        self.best_action = best_action
        self.bound = bound
        self.depth = depth

class MCTSTranspositionTable:
    """Converts MCTS tree into a DAG. Same position, pooled Q/N statistics."""
    def __init__(self):
        self.table = {}

    def get_or_create(self, board_hash: int, board_state) -> Tuple[Any, bool]:
        if board_hash in self.table:
            return self.table[board_hash], True
        
        from .mcts import MCTSNode
        node = MCTSNode(board_state)
        self.table[board_hash] = node
        return node, False

    def clear(self):
        self.table.clear()


class SolverTranspositionTable:
    """Stores alpha-beta bounds for MTD(f)."""
    def __init__(self):
        self.table: Dict[int, TTEntry] = {}

    def probe(self, board_hash: int, alpha: float, beta: float, depth: int) -> Tuple[float, Any]:
        entry = self.table.get(board_hash)
        if entry is None or entry.depth < depth:
            return None, None
        
        if entry.bound == BoundType.EXACT:
            return entry.value, entry.best_action
        if entry.bound == BoundType.LOWER and entry.value >= beta:
            return entry.value, entry.best_action
        if entry.bound == BoundType.UPPER and entry.value <= alpha:
            return entry.value, entry.best_action
            
        return None, None

    def store(self, board_hash: int, value: float, best_action: Any, alpha_orig: float, beta: float, depth: int):
        if value <= alpha_orig:
            bound = BoundType.UPPER
        elif value >= beta:
            bound = BoundType.LOWER
        else:
            bound = BoundType.EXACT
            
        self.table[board_hash] = TTEntry(value, best_action, bound, depth)

    def clear(self):
        self.table.clear()
