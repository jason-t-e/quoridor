import numpy as np
from engine.board import BoardState
from engine.moves import Move, PawnMove, WallMove
from engine.pathfinder import bfs_distance

def encode_board(board: BoardState) -> np.ndarray:
    """ Converts BoardState to 10x9x9 float32 tensor. """
    tensor = np.zeros((10, 9, 9), dtype=np.float32)
    p = board.current_player
    opp = 1 - p
    
    r, c = board.pawn_positions[p]
    tensor[0, r, c] = 1.0
    
    orow, ocol = board.pawn_positions[opp]
    tensor[1, orow, ocol] = 1.0
    
    for hr, hc in board.h_walls: tensor[2, hr, hc] = 1.0
    for vr, vc in board.v_walls: tensor[3, vr, vc] = 1.0
        
    own_dist = bfs_distance(board, p)
    opp_dist = bfs_distance(board, opp)
    tensor[4, :, :] = own_dist / 100.0
    tensor[5, :, :] = opp_dist / 100.0
    
    tensor[6, :, :] = board.walls_remaining[p] / 10.0
    tensor[7, :, :] = board.walls_remaining[opp] / 10.0
    
    phase = min(1.0, board.turn / 60.0)
    tensor[8, :, :] = phase
    tensor[9, :, :] = 1.0 if p == 0 else -1.0
    
    if p == 1:
        # Flip perspective vertically for player 1
        tensor = tensor[:, ::-1, :].copy()
    
    return tensor

def encode_action(board: BoardState, move: Move) -> int:
    """ Maps a move to index [0, 135] """
    if isinstance(move, WallMove):
        # If player 1, flip the row perspective for walls as well
        r = move.row if board.current_player == 0 else 7 - move.row
        idx = r * 8 + move.col
        if move.orientation == 'v':
            idx += 64
        return idx
    elif isinstance(move, PawnMove):
        p = board.current_player
        start_r, start_c = board.pawn_positions[p]
        dr = move.to_row - start_r
        dc = move.to_col - start_c
        
        if p == 1:
            dr = -dr
            
        if dr < 0 and dc == 0: return 128
        elif dr > 0 and dc == 0: return 129
        elif dr == 0 and dc > 0: return 130
        elif dr == 0 and dc < 0: return 131
        elif dr < 0 and dc > 0: return 132
        elif dr < 0 and dc < 0: return 133
        elif dr > 0 and dc > 0: return 134
        elif dr > 0 and dc < 0: return 135
        else: return 128
    return 0

def decode_action(board: BoardState, idx: int) -> Move:
    if idx < 64:
        r, c = divmod(idx, 8)
        if board.current_player == 1: r = 7 - r
        return WallMove(r, c, 'h')
    elif idx < 128:
        r, c = divmod(idx - 64, 8)
        if board.current_player == 1: r = 7 - r
        return WallMove(r, c, 'v')
    else:
        p = board.current_player
        start_r, start_c = board.pawn_positions[p]
        direction = idx - 128
        d_map = [(-1,0), (1,0), (0,1), (0,-1), (-1,1), (-1,-1), (1,1), (1,-1)]
        dr, dc = d_map[direction]
        if p == 1: dr = -dr
        
        from engine.rules import get_pawn_moves
        legal = get_pawn_moves(board)
        for lm in legal:
            ldr = lm.to_row - start_r
            ldc = lm.to_col - start_c
            ndr = -1 if ldr < 0 else (1 if ldr > 0 else 0)
            ndc = -1 if ldc < 0 else (1 if ldc > 0 else 0)
            if ndr == dr and ndc == dc:
                return lm
        return PawnMove(start_r + dr, start_c + dc)
