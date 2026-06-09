from engine.board import BoardState
from engine.moves import WallMove
from engine.rules import is_valid_wall_move

def test_valid_wall_placement():
    board = BoardState()
    move = WallMove(4, 4, 'h')
    assert is_valid_wall_move(board, move) == True

def test_wall_overlap_same_wall():
    board = BoardState()
    board.h_walls.add((4, 4))
    move = WallMove(4, 4, 'h')
    assert is_valid_wall_move(board, move) == False

def test_wall_overlap_adjacent():
    board = BoardState()
    board.h_walls.add((4, 4))
    
    # Half-overlap left
    move1 = WallMove(4, 3, 'h')
    assert is_valid_wall_move(board, move1) == False
    
    # Half-overlap right
    move2 = WallMove(4, 5, 'h')
    assert is_valid_wall_move(board, move2) == False

def test_wall_crossing():
    board = BoardState()
    board.h_walls.add((4, 4))
    
    # A vertical wall at the same coordinate crosses the horizontal wall exactly in the center
    move = WallMove(4, 4, 'v')
    assert is_valid_wall_move(board, move) == False

def test_out_of_bounds_wall():
    board = BoardState()
    assert is_valid_wall_move(board, WallMove(8, 4, 'h')) == False
    assert is_valid_wall_move(board, WallMove(4, 8, 'v')) == False
    assert is_valid_wall_move(board, WallMove(-1, 0, 'h')) == False

def test_no_walls_remaining():
    board = BoardState()
    board.walls_remaining[0] = 0
    board.current_player = 0
    assert is_valid_wall_move(board, WallMove(4, 4, 'h')) == False
