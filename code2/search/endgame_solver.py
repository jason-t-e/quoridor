from game_engine.rules import get_legal_moves

class EndgameSolver:
    def solve(self, board, solver_tt):
        """MTD(f) exact solver"""
        f, best_move = 0, None
        
        max_depth = 15 # Example depth limit
        
        for depth in range(1, max_depth + 1):
            value, move = self._mtdf(board, f, depth, solver_tt)
            f, best_move = value, move
            if abs(value) >= 0.9: # 1.0 is win
                break
        return best_move

    def _mtdf(self, board, f, depth, tt):
        g = f
        upperbound = 1.0
        lowerbound = -1.0
        best_move = None
        
        while lowerbound < upperbound:
            beta = max(g, lowerbound + 0.01)
            g, move = self._alpha_beta(board, beta - 0.01, beta, depth, tt)
            if best_move is None:
                best_move = move
            if g < beta:
                upperbound = g
            else:
                lowerbound = g
                best_move = move
        return g, best_move

    def _alpha_beta(self, board, alpha, beta, depth, tt):
        if board.is_terminal():
            return (1.0 if board.winner == board.current_player else -1.0), None
            
        if depth == 0:
            return 0.0, None # Eval
            
        tt_val, tt_move = tt.probe(board.current_hash, alpha, beta, depth)
        if tt_val is not None:
            return tt_val, tt_move
            
        best_value = -float('inf')
        best_move = None
        
        alpha_orig = alpha
        
        moves = get_legal_moves(board)
        for move in moves:
            child = board.apply_move(move)
            val, _ = self._alpha_beta(child, -beta, -alpha, depth - 1, tt)
            val = -val
            
            if val > best_value:
                best_value = val
                best_move = move
                
            alpha = max(alpha, val)
            if alpha >= beta:
                break
                
        tt.store(board.current_hash, best_value, best_move, alpha_orig, beta, depth)
        return best_value, best_move
