from engine.board import BoardState
from engine.moves import WallMove
from engine.rules import is_valid_wall_move

def test_path_existence():
    # Enclose player 0
    board = BoardState()
    board.pawn_positions[0] = (0, 0)
    
    # Add walls to block the right and bottom completely except for one hole
    board.v_walls.add((0, 0))
    # Trying to add horizontal wall to completely block
    # Player 0 at (0,0) wants to reach row 8.
    
    move = WallMove(0, 0, 'h')
    
    # This should be invalid because it traps player 0
    assert is_valid_wall_move(board, move) == False

def test_path_existence_complex():
    board = BoardState()
    # Build a wall completely across row 2, except for one gap
    for c in range(7):
        board.h_walls.add((2, c))
        
    # The only gap is at c=7 and c=8 (which means wall placed at 7 spans 7 and 8)
    # If we try to place the last wall at (2,7), it blocks the board completely
    move = WallMove(2, 7, 'h')
    
    assert is_valid_wall_move(board, move) == False
    
    # Placing it elsewhere should be fine
    move2 = WallMove(3, 7, 'h')
    assert is_valid_wall_move(board, move2) == True
