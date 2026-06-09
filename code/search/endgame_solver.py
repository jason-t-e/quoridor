"""
Endgame Solver for Quoridor.

Uses MTD(f) with alpha-beta search and a solver Transposition Table
to find exact play when total walls remaining ≤ threshold (default 4).

The solver is triggered automatically by MCTS and the GUI when the
endgame condition is met, providing perfect play in simplified positions.
"""

import math
from engine.game import QuoridorGame
from engine.rules import get_all_legal_moves
from engine.pathfinder import bfs_distance
from engine.moves import PawnMove, WallMove

# TT entry flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


class EndgameSolver:
    def __init__(self, wall_threshold: int = 4, max_depth: int = 40):
        self.wall_threshold = wall_threshold
        self.max_depth = max_depth
        self.solver_tt = {}   # zobrist_hash -> (score, flag, depth, best_move)
        self.nodes_searched = 0

    def should_activate(self, board) -> bool:
        """Check if endgame solver should take over."""
        total_walls = sum(board.walls_remaining.values())
        return total_walls <= self.wall_threshold

    def solve(self, board, player: int):
        """
        Find the best move using MTD(f).
        Returns (score, best_move).
        """
        self.nodes_searched = 0
        first_guess = self._heuristic(board, player)

        # Iterative deepening MTD(f)
        best_move = None
        score = first_guess
        for depth in range(2, self.max_depth + 1, 2):
            score, best_move = self._mtdf(board, score, depth, player)
            # If we found a winning/losing terminal, no need to go deeper
            if abs(score) >= 900:
                break

        return score, best_move

    def _mtdf(self, board, f, depth, player):
        """MTD(f) driver — performs null-window alpha-beta searches."""
        g = f
        upper = math.inf
        lower = -math.inf
        best_move = None

        while lower < upper:
            beta = max(g, lower + 1)
            g, best_move = self._alphabeta_tt(
                board, depth, beta - 1, beta, True, player)
            if g < beta:
                upper = g
            else:
                lower = g

        return g, best_move

    def _alphabeta_tt(self, board, depth, alpha, beta, maximizing, root_player):
        """Alpha-beta with Transposition Table, move ordering, and terminal detection."""
        self.nodes_searched += 1

        # Terminal check
        p0_goal = board.pawn_positions[0][0] == 8
        p1_goal = board.pawn_positions[1][0] == 0

        if p0_goal:
            return (1000 if root_player == 0 else -1000), None
        if p1_goal:
            return (1000 if root_player == 1 else -1000), None

        if depth == 0:
            return self._heuristic(board, root_player), None

        # TT lookup
        tt_key = board.current_hash
        tt_entry = self.solver_tt.get(tt_key)
        tt_move = None

        if tt_entry is not None:
            tt_score, tt_flag, tt_depth, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == EXACT:
                    return tt_score, tt_move
                elif tt_flag == LOWERBOUND:
                    alpha = max(alpha, tt_score)
                elif tt_flag == UPPERBOUND:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score, tt_move

        moves = get_all_legal_moves(board)
        if not moves:
            return self._heuristic(board, root_player), None

        # Move ordering: TT move first, then pawn moves sorted by distance improvement
        moves = self._order_moves(board, moves, tt_move, root_player, maximizing)

        best_move = moves[0]
        original_alpha = alpha

        if maximizing:
            max_eval = -math.inf
            for move in moves:
                game = QuoridorGame()
                game.board = board.clone()
                game.apply_move(move)

                eval_score, _ = self._alphabeta_tt(
                    game.board, depth - 1, alpha, beta, False, root_player)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            # Store in TT
            if max_eval <= original_alpha:
                flag = UPPERBOUND
            elif max_eval >= beta:
                flag = LOWERBOUND
            else:
                flag = EXACT
            self.solver_tt[tt_key] = (max_eval, flag, depth, best_move)
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                game = QuoridorGame()
                game.board = board.clone()
                game.apply_move(move)

                eval_score, _ = self._alphabeta_tt(
                    game.board, depth - 1, alpha, beta, True, root_player)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            # Store in TT
            if min_eval >= beta:
                flag = LOWERBOUND  # Was a cut node for the opponent
            elif min_eval <= original_alpha:
                flag = UPPERBOUND
            else:
                flag = EXACT
            self.solver_tt[tt_key] = (min_eval, flag, depth, best_move)
            return min_eval, best_move

    def _heuristic(self, board, player) -> float:
        """Path-distance differential heuristic."""
        own_dist = bfs_distance(board, player)
        opp_dist = bfs_distance(board, 1 - player)
        return float(opp_dist - own_dist)

    def _order_moves(self, board, moves, tt_move, root_player, maximizing):
        """
        Order moves for better alpha-beta pruning:
        1. TT best move first
        2. Pawn moves sorted by distance improvement
        3. Wall moves last
        """
        current_player = board.current_player

        tt_match = None
        pawn_moves = []
        wall_moves = []

        for m in moves:
            if tt_move is not None and self._moves_equal(m, tt_move):
                tt_match = m
                continue
            if isinstance(m, PawnMove):
                pawn_moves.append(m)
            else:
                wall_moves.append(m)

        # Sort pawn moves by how much they reduce our distance to goal
        current_dist = bfs_distance(board, current_player)

        def pawn_key(m):
            # Simulate move and measure new distance
            game = QuoridorGame()
            game.board = board.clone()
            game.apply_move(m)
            new_dist = bfs_distance(game.board, current_player)
            return new_dist  # Lower is better

        pawn_moves.sort(key=pawn_key)

        ordered = []
        if tt_match is not None:
            ordered.append(tt_match)
        ordered.extend(pawn_moves)
        ordered.extend(wall_moves)

        return ordered

    @staticmethod
    def _moves_equal(a, b) -> bool:
        """Compare two Move objects for equality."""
        if type(a) != type(b):
            return False
        if isinstance(a, PawnMove):
            return a.to_row == b.to_row and a.to_col == b.to_col
        if isinstance(a, WallMove):
            return (a.row == b.row and a.col == b.col
                    and a.orientation == b.orientation)
        return False


# Singleton for reuse across a game
_solver_instance = None


def get_solver(wall_threshold: int = 4) -> EndgameSolver:
    """Get or create a shared EndgameSolver instance."""
    global _solver_instance
    if _solver_instance is None or _solver_instance.wall_threshold != wall_threshold:
        _solver_instance = EndgameSolver(wall_threshold=wall_threshold)
    return _solver_instance
