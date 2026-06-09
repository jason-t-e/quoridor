from typing import List
from engine.board import BoardState
from engine.moves import Move, PawnMove, WallMove
from engine.pathfinder import get_valid_neighbors, has_path_to_goal, is_blocked_up, is_blocked_down, is_blocked_left, is_blocked_right

def is_wall_overlap(board: BoardState, move: WallMove) -> bool:
    r, c = move.row, move.col
    if move.orientation == 'h':
        # Cannot overlap exactly or overlap with adjacent horizontal walls
        if (r, c) in board.h_walls: return True
        if (r, c - 1) in board.h_walls: return True
        if (r, c + 1) in board.h_walls: return True
        # Cannot cross a vertical wall at the same center
        if (r, c) in board.v_walls: return True
    else:
        # Cannot overlap exactly or overlap with adjacent vertical walls
        if (r, c) in board.v_walls: return True
        if (r - 1, c) in board.v_walls: return True
        if (r + 1, c) in board.v_walls: return True
        # Cannot cross a horizontal wall at the same center
        if (r, c) in board.h_walls: return True
    return False

def is_valid_wall_move(board: BoardState, move: WallMove) -> bool:
    # Check boundary
    if move.row < 0 or move.row > 7 or move.col < 0 or move.col > 7:
        return False
        
    # Check wall count
    if board.walls_remaining[board.current_player] <= 0:
        return False
        
    # Check overlap
    if is_wall_overlap(board, move):
        return False
        
    # Check anti-blockade
    temp_board = board.clone()
    if move.orientation == 'h':
        temp_board.h_walls.add((move.row, move.col))
    else:
        temp_board.v_walls.add((move.row, move.col))
        
    if not has_path_to_goal(temp_board, 0) or not has_path_to_goal(temp_board, 1):
        return False
        
    return True

def get_pawn_moves(board: BoardState) -> List[PawnMove]:
    moves = []
    p = board.current_player
    opp = 1 - p
    r, c = board.pawn_positions[p]
    opp_r, opp_c = board.pawn_positions[opp]
    
    # Check all 4 basic directions
    for dr, dc, is_blocked_func in [(-1, 0, is_blocked_up), (1, 0, is_blocked_down), (0, -1, is_blocked_left), (0, 1, is_blocked_right)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr <= 8 and 0 <= nc <= 8 and not is_blocked_func(board, r, c):
            if (nr, nc) == (opp_r, opp_c):
                # Opponent is in the way. Can we jump over?
                jump_r, jump_c = nr + dr, nc + dc
                if 0 <= jump_r <= 8 and 0 <= jump_c <= 8 and not is_blocked_func(board, nr, nc):
                    # Direct jump
                    moves.append(PawnMove(jump_r, jump_c))
                else:
                    # Direct jump is blocked by boundary or wall. Can we jump diagonally?
                    # Check perpendicular directions from the opponent's position
                    if dr != 0: # We moved vertically
                        # Check left from opp
                        if nc > 0 and not is_blocked_left(board, opp_r, opp_c):
                            moves.append(PawnMove(opp_r, opp_c - 1))
                        # Check right from opp
                        if nc < 8 and not is_blocked_right(board, opp_r, opp_c):
                            moves.append(PawnMove(opp_r, opp_c + 1))
                    else: # We moved horizontally
                        # Check up from opp
                        if opp_r > 0 and not is_blocked_up(board, opp_r, opp_c):
                            moves.append(PawnMove(opp_r - 1, opp_c))
                        # Check down from opp
                        if opp_r < 8 and not is_blocked_down(board, opp_r, opp_c):
                            moves.append(PawnMove(opp_r + 1, opp_c))
            else:
                # Normal move
                moves.append(PawnMove(nr, nc))
                
    return moves

def get_all_legal_moves(board: BoardState) -> List[Move]:
    moves = []
    
    # Add pawn moves
    moves.extend(get_pawn_moves(board))
    
    # Add wall moves
    if board.walls_remaining[board.current_player] > 0:
        for r in range(8):
            for c in range(8):
                h_move = WallMove(r, c, 'h')
                if is_valid_wall_move(board, h_move):
                    moves.append(h_move)
                    
                v_move = WallMove(r, c, 'v')
                if is_valid_wall_move(board, v_move):
                    moves.append(v_move)
                    
    return moves
