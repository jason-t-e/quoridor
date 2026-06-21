import pytest
from game_engine.board import BoardState
from game_engine.moves import PawnMove, WallMove
from game_engine.rules import get_legal_moves
from game_engine.zobrist_hash import HASHER

def test_initial_board():
    board = BoardState()
    assert board.pawn_positions[0] == (0, 4)
    assert board.pawn_positions[1] == (8, 4)
    assert board.walls_remaining[0] == 10
    assert board.walls_remaining[1] == 10
    assert board.current_player == 0

def test_zobrist_incremental_vs_full():
    board = BoardState()
    initial_hash = board.current_hash
    assert initial_hash == HASHER.full_hash(board)
    
    # Apply a pawn move
    move = PawnMove(player=0, to_row=1, to_col=4)
    new_board = board.apply_move(move)
    
    # Check if incremental matches full
    assert new_board.current_hash == HASHER.full_hash(new_board)
    assert new_board.current_hash != initial_hash
    
    # Apply a wall move
    wall_move = WallMove(player=1, row=4, col=4, orientation='h')
    new_board2 = new_board.apply_move(wall_move)
    
    assert new_board2.current_hash == HASHER.full_hash(new_board2)

def test_anti_blockade():
    board = BoardState()
    # To block player 0 (starting at 0,4) from going down,
    # we can build a wall across row 1: (1,0), (1,2), (1,4), (1,6) 'h'
    # This leaves only column 8 open.
    # To block column 8, we can put a 'v' wall at (0,7), which blocks
    # moving from (0,7) to (0,8) and from (1,7) to (1,8).
    board = board.apply_move(WallMove(0, 1, 0, 'h'))
    board = board.apply_move(WallMove(0, 1, 2, 'h'))
    board = board.apply_move(WallMove(0, 1, 4, 'h'))
    board = board.apply_move(WallMove(0, 1, 6, 'h'))
    
    from game_engine.rules import is_valid_wall
    blocking_wall = WallMove(0, 0, 7, 'v')
    assert is_valid_wall(board, blocking_wall) == False
    
def test_board_to_tensor():
    from game_engine.board import board_to_tensor
    board = BoardState()
    tensor = board_to_tensor(board)
    assert tensor.shape == (10, 9, 9)
