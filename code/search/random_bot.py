import random
from engine.rules import get_all_legal_moves

def get_random_move(board):
    moves = get_all_legal_moves(board)
    if not moves:
        return None
    return random.choice(moves)
