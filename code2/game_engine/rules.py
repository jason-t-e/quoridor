from .board import BoardState
from .moves import Move, PawnMove, WallMove
from .pathfinder import shortest_path_to_goal, get_valid_neighbors

def is_wall_overlap(board: BoardState, move: WallMove) -> bool:
    r, c = move.row, move.col
    if move.orientation == 'h':
        # Exact overlap
        if (r, c) in board.h_walls: return True
        # Shifted overlap
        if (r, c-1) in board.h_walls or (r, c+1) in board.h_walls: return True
        # Cross overlap
        if (r, c) in board.v_walls: return True
    else:
        # Exact overlap
        if (r, c) in board.v_walls: return True
        # Shifted overlap
        if (r-1, c) in board.v_walls or (r+1, c) in board.v_walls: return True
        # Cross overlap
        if (r, c) in board.h_walls: return True
    return False

def is_valid_wall(board: BoardState, move: WallMove) -> bool:
    if board.walls_remaining[move.player] <= 0:
        return False
    if not (0 <= move.row < 8 and 0 <= move.col < 8):
        return False
    if is_wall_overlap(board, move):
        return False
        
    # Anti-blockade check
    # We simulate applying the wall, and check shortest paths
    temp_board = board.apply_move(move)
    if shortest_path_to_goal(temp_board, 0) == 999: return False
    if shortest_path_to_goal(temp_board, 1) == 999: return False
    
    return True

def get_legal_pawn_moves(board: BoardState) -> list[PawnMove]:
    p = board.current_player
    opp = 1 - p
    r, c = board.pawn_positions[p]
    opp_r, opp_c = board.pawn_positions[opp]
    
    moves = []
    # Basic neighbors (ignoring opponent)
    valid_neighbors = get_valid_neighbors(r, c, board)
    
    for nr, nc in valid_neighbors:
        if (nr, nc) == (opp_r, opp_c):
            # Jump logic
            opp_neighbors = get_valid_neighbors(opp_r, opp_c, board)
            
            # Straight jump
            dr = opp_r - r
            dc = opp_c - c
            jump_r = opp_r + dr
            jump_c = opp_c + dc
            
            if (jump_r, jump_c) in opp_neighbors:
                moves.append(PawnMove(player=p, to_row=jump_r, to_col=jump_c))
            else:
                # Diagonal jump because straight is blocked by a wall or board edge
                for d_nr, d_nc in opp_neighbors:
                    if (d_nr, d_nc) != (r, c):
                        moves.append(PawnMove(player=p, to_row=d_nr, to_col=d_nc))
        else:
            moves.append(PawnMove(player=p, to_row=nr, to_col=nc))
            
    return moves

def get_legal_wall_moves(board: BoardState) -> list[WallMove]:
    p = board.current_player
    if board.walls_remaining[p] == 0:
        return []
        
    moves = []
    for r in range(8):
        for c in range(8):
            wm_h = WallMove(player=p, row=r, col=c, orientation='h')
            if not is_wall_overlap(board, wm_h):
                temp = board.apply_move(wm_h)
                if shortest_path_to_goal(temp, 0) != 999 and shortest_path_to_goal(temp, 1) != 999:
                    moves.append(wm_h)
                    
            wm_v = WallMove(player=p, row=r, col=c, orientation='v')
            if not is_wall_overlap(board, wm_v):
                temp = board.apply_move(wm_v)
                if shortest_path_to_goal(temp, 0) != 999 and shortest_path_to_goal(temp, 1) != 999:
                    moves.append(wm_v)
                    
    return moves

def get_legal_moves(board: BoardState) -> list[Move]:
    return get_legal_pawn_moves(board) + get_legal_wall_moves(board)
