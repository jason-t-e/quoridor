import random
from typing import List
from engine.board import BoardState
from engine.moves import Move, PawnMove, WallMove
from engine.rules import get_all_legal_moves
from engine import zobrist

class QuoridorGame:
    def __init__(self):
        self.board = BoardState()
        self.winner = None
        self.is_terminal = False

    def get_legal_moves(self) -> List[Move]:
        return get_all_legal_moves(self.board)

    def apply_move(self, move: Move):
        if self.is_terminal:
            raise ValueError("Game is already over")
            
        p = self.board.current_player
        
        if isinstance(move, PawnMove):
            old_pos = self.board.pawn_positions[p]
            new_pos = (move.to_row, move.to_col)
            self.board.pawn_positions[p] = new_pos
            # Incremental Zobrist update for pawn move
            self.board.current_hash = zobrist.incremental_pawn_move(
                self.board.current_hash, p, old_pos, new_pos
            )
        elif isinstance(move, WallMove):
            old_wall_count = self.board.walls_remaining[p]
            if move.orientation == 'h':
                self.board.h_walls.add((move.row, move.col))
            else:
                self.board.v_walls.add((move.row, move.col))
            self.board.walls_remaining[p] -= 1
            # Incremental Zobrist update for wall placement
            self.board.current_hash = zobrist.incremental_wall_place(
                self.board.current_hash, p, move.orientation,
                (move.row, move.col), old_wall_count
            )
            
        self.board.move_history.append(move)
        
        # Check termination
        if (p == 0 and self.board.pawn_positions[p][0] == 8) or \
           (p == 1 and self.board.pawn_positions[p][0] == 0):
            self.is_terminal = True
            self.winner = p
            
        self.board.current_player = 1 - p
        self.board.turn += 1
        # Toggle side-to-move in hash
        self.board.current_hash = zobrist.incremental_turn(self.board.current_hash)

    def play_random_game(self):
        while not self.is_terminal:
            moves = self.get_legal_moves()
            if not moves:
                raise RuntimeError("No legal moves available, but game is not over.")
            move = random.choice(moves)
            self.apply_move(move)
        return self.winner

