from collections import deque
from typing import Tuple, List, Set
from engine.board import BoardState

def is_blocked_up(board: BoardState, r: int, c: int) -> bool:
    return (r - 1, c) in board.h_walls or (r - 1, c - 1) in board.h_walls

def is_blocked_down(board: BoardState, r: int, c: int) -> bool:
    return (r, c) in board.h_walls or (r, c - 1) in board.h_walls

def is_blocked_left(board: BoardState, r: int, c: int) -> bool:
    return (r, c - 1) in board.v_walls or (r - 1, c - 1) in board.v_walls

def is_blocked_right(board: BoardState, r: int, c: int) -> bool:
    return (r, c) in board.v_walls or (r - 1, c) in board.v_walls

def get_valid_neighbors(board: BoardState, r: int, c: int) -> List[Tuple[int, int]]:
    neighbors = []
    # Up
    if r > 0 and not is_blocked_up(board, r, c):
        neighbors.append((r - 1, c))
    # Down
    if r < 8 and not is_blocked_down(board, r, c):
        neighbors.append((r + 1, c))
    # Left
    if c > 0 and not is_blocked_left(board, r, c):
        neighbors.append((r, c - 1))
    # Right
    if c < 8 and not is_blocked_right(board, r, c):
        neighbors.append((r, c + 1))
    return neighbors

def has_path_to_goal(board: BoardState, player: int) -> bool:
    start = board.pawn_positions[player]
    target_row = 8 if player == 0 else 0
    
    visited = {start}
    queue = deque([start])
    
    while queue:
        r, c = queue.popleft()
        if r == target_row:
            return True
            
        for nr, nc in get_valid_neighbors(board, r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
                
    return False

def bfs_distance(board: BoardState, player: int) -> int:
    start = board.pawn_positions[player]
    target_row = 8 if player == 0 else 0
    
    visited = {start}
    queue = deque([(start[0], start[1], 0)])
    
    while queue:
        r, c, dist = queue.popleft()
        if r == target_row:
            return dist
            
        for nr, nc in get_valid_neighbors(board, r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
                
    return 999  # Unreachable (should not happen with valid rules)
