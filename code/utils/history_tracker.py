import json
import os
from engine.board import BoardState
from engine.moves import Move

class HistoryTracker:
    def __init__(self, filepath="data/global_history.json"):
        self.filepath = filepath
        self.graph = {}
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.graph = json.load(f)
            except Exception as e:
                # If there's an error reading (e.g. empty file), initialize empty
                self.graph = {}
        else:
            self.graph = {}

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.graph, f, indent=2)
        except Exception as e:
            print(f"Error saving global history: {e}")

    def get_state_key(self, board: BoardState) -> str:
        # Generate a unique canonical representation for the state
        p0 = board.pawn_positions[0]
        p1 = board.pawn_positions[1]
        w0 = board.walls_remaining[0]
        w1 = board.walls_remaining[1]
        hw = sorted(list(board.h_walls))
        vw = sorted(list(board.v_walls))
        
        hw_str = ",".join(f"{r}_{c}" for r, c in hw)
        vw_str = ",".join(f"{r}_{c}" for r, c in vw)
        
        return f"P0:{p0[0]}_{p0[1]}|P1:{p1[0]}_{p1[1]}|W0:{w0}|W1:{w1}|H:[{hw_str}]|V:[{vw_str}]|Player:{board.current_player}"

    def record_game_step(self, board_before: BoardState, move: Move, board_after: BoardState):
        key_before = self.get_state_key(board_before)
        key_after = self.get_state_key(board_after)
        move_str = repr(move)

        # Ensure source state node exists
        if key_before not in self.graph:
            self.graph[key_before] = {
                "pawn_positions": {"0": list(board_before.pawn_positions[0]), "1": list(board_before.pawn_positions[1])},
                "walls_remaining": {"0": board_before.walls_remaining[0], "1": board_before.walls_remaining[1]},
                "h_walls": [list(wall) for wall in board_before.h_walls],
                "v_walls": [list(wall) for wall in board_before.v_walls],
                "current_player": board_before.current_player,
                "visit_count": 0,
                "winner_counts": {"0": 0, "1": 0},
                "moves": {}
            }
        
        # Ensure destination state node exists
        if key_after not in self.graph:
            self.graph[key_after] = {
                "pawn_positions": {"0": list(board_after.pawn_positions[0]), "1": list(board_after.pawn_positions[1])},
                "walls_remaining": {"0": board_after.walls_remaining[0], "1": board_after.walls_remaining[1]},
                "h_walls": [list(wall) for wall in board_after.h_walls],
                "v_walls": [list(wall) for wall in board_after.v_walls],
                "current_player": board_after.current_player,
                "visit_count": 0,
                "winner_counts": {"0": 0, "1": 0},
                "moves": {}
            }

        # Increment visit count for the starting state
        self.graph[key_before]["visit_count"] += 1
        
        # Record the transition
        moves_dict = self.graph[key_before]["moves"]
        if move_str not in moves_dict:
            moves_dict[move_str] = {
                "next_state_key": key_after,
                "play_count": 0
            }
        moves_dict[move_str]["play_count"] += 1

    def record_game_outcome(self, path: list, winner: int):
        # path is a sequence of state keys visited during a single game
        if winner is None or winner not in (0, 1):
            return
        winner_str = str(winner)
        for key in path:
            if key in self.graph:
                if "winner_counts" not in self.graph[key]:
                    self.graph[key]["winner_counts"] = {"0": 0, "1": 0}
                self.graph[key]["winner_counts"][winner_str] += 1
