from engine.board import BoardState
from engine.moves import PawnMove, WallMove
from engine.rules import get_pawn_moves

def test_pawn_forward_movement():
    board = BoardState()
    # Player 0 is at (0,4)
    moves = get_pawn_moves(board)
    assert PawnMove(1, 4) in moves
    assert PawnMove(0, 3) in moves
    assert PawnMove(0, 5) in moves
    # Player 0 cannot move up since row is 0
    assert PawnMove(-1, 4) not in moves

def test_pawn_jump_movement():
    board = BoardState()
    board.pawn_positions[0] = (4, 4)
    board.pawn_positions[1] = (5, 4)
    board.current_player = 0
    
    moves = get_pawn_moves(board)
    # Player 0 should be able to jump over Player 1 to (6, 4)
    assert PawnMove(6, 4) in moves
    assert PawnMove(5, 4) not in moves

def test_pawn_diagonal_jump():
    board = BoardState()
    board.pawn_positions[0] = (4, 4)
    board.pawn_positions[1] = (5, 4)
    board.current_player = 0
    
    # Place a horizontal wall behind player 1, blocking direct jump
    board.h_walls.add((5, 4))
    
    moves = get_pawn_moves(board)
    # Direct jump to (6,4) is blocked. Should be able to jump diagonally to (5,3) and (5,5)
    assert PawnMove(6, 4) not in moves
    assert PawnMove(5, 3) in moves
    assert PawnMove(5, 5) in moves
    
    # Also normal moves
    assert PawnMove(3, 4) in moves
    assert PawnMove(4, 3) in moves
    assert PawnMove(4, 5) in moves

def test_pawn_diagonal_jump_blocked_by_wall():
    board = BoardState()
    board.pawn_positions[0] = (4, 4)
    board.pawn_positions[1] = (5, 4)
    board.current_player = 0
    
    board.h_walls.add((5, 4)) # Block forward jump
    board.v_walls.add((5, 4)) # Block diagonal jump to the right (5, 5)
    
    moves = get_pawn_moves(board)
    assert PawnMove(5, 5) not in moves
    assert PawnMove(5, 3) in moves
