import numpy as np
from collections import deque
from .board import BoardState

def get_valid_neighbors(r: int, c: int, board: BoardState) -> list[tuple[int, int]]:
    neighbors = []
    # Up
    if r > 0 and not ((r-1, c) in board.h_walls or (r-1, c-1) in board.h_walls):
        neighbors.append((r-1, c))
    # Down
    if r < 8 and not ((r, c) in board.h_walls or (r, c-1) in board.h_walls):
        neighbors.append((r+1, c))
    # Left
    if c > 0 and not ((r, c-1) in board.v_walls or (r-1, c-1) in board.v_walls):
        neighbors.append((r, c-1))
    # Right
    if c < 8 and not ((r, c) in board.v_walls or (r-1, c) in board.v_walls):
        neighbors.append((r, c+1))
    return neighbors

def bfs_distance_map(board: BoardState, player: int) -> np.ndarray:
    dist = np.full((9, 9), 999, dtype=np.float32)
    start_r, start_c = board.pawn_positions[player]
    
    queue = deque([(start_r, start_c)])
    dist[start_r, start_c] = 0
    
    while queue:
        r, c = queue.popleft()
        d = dist[r, c]
        for nr, nc in get_valid_neighbors(r, c, board):
            if dist[nr, nc] > d + 1:
                dist[nr, nc] = d + 1
                queue.append((nr, nc))
                
    return dist

def shortest_path_to_goal(board: BoardState, player: int) -> int:
    start_r, start_c = board.pawn_positions[player]
    goal_row = 8 if player == 0 else 0
    
    if start_r == goal_row:
        return 0
        
    queue = deque([(start_r, start_c, 0)])
    visited = {(start_r, start_c)}
    
    while queue:
        r, c, d = queue.popleft()
        for nr, nc in get_valid_neighbors(r, c, board):
            if nr == goal_row:
                return d + 1
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, d + 1))
                
    return 999 # blocked

def path_differential(board: BoardState) -> int:
    p = board.current_player
    opp = 1 - p
    return shortest_path_to_goal(board, opp) - shortest_path_to_goal(board, p)
