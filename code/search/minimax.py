import math
from engine.game import QuoridorGame
from engine.rules import get_all_legal_moves
from engine.pathfinder import bfs_distance

def evaluate_state(board, player):
    # Path differential: Positive if we are closer to goal than opponent
    own_dist = bfs_distance(board, player)
    opp_dist = bfs_distance(board, 1 - player)
    return opp_dist - own_dist

def alphabeta(board, depth, alpha, beta, maximizing_player, root_player):
    p0_goal = board.pawn_positions[0][0] == 8
    p1_goal = board.pawn_positions[1][0] == 0
    
    if p0_goal:
        return (1000 if root_player == 0 else -1000), None
    if p1_goal:
        return (1000 if root_player == 1 else -1000), None
        
    if depth == 0:
        return evaluate_state(board, root_player), None
        
    from engine.moves import PawnMove
    moves = get_all_legal_moves(board)
    if not moves:
        return evaluate_state(board, root_player), None
        
    # Focus on pawn moves at deeper search ply to keep lookahead extremely fast
    if depth <= 6:
        pawn_moves = [m for m in moves if isinstance(m, PawnMove)]
        if pawn_moves:
            moves = pawn_moves
            
    # Sort PawnMove first to maximize alpha-beta cutoffs
    moves = sorted(moves, key=lambda m: 0 if isinstance(m, PawnMove) else 1)
    
    best_move = moves[0]
    
    if maximizing_player:
        max_eval = -math.inf
        for move in moves:
            game = QuoridorGame()
            game.board = board.clone()
            game.apply_move(move)
            
            eval_score, _ = alphabeta(game.board, depth - 1, alpha, beta, False, root_player)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            game = QuoridorGame()
            game.board = board.clone()
            game.apply_move(move)
            
            eval_score, _ = alphabeta(game.board, depth - 1, alpha, beta, True, root_player)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move
