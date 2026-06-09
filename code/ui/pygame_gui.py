import pygame
import sys
import os
import threading
import torch
import torch.optim as optim
import numpy as np
from typing import Optional

from engine.game import QuoridorGame
from engine.moves import Move, PawnMove, WallMove
from training.trainer import train_step
from training.experience_buffer import ExperienceBuffer
from models.quoridor_net import QuoridorNet, UniformModel
from models.board_encoder import encode_board, decode_action
from search.mcts import MCTS
from search.endgame_solver import get_solver
from utils.history_tracker import HistoryTracker
from training.model_registry import load_metadata, BEST_MODEL_PATH

# Constants
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750
BOARD_OFFSET_X = 50
BOARD_OFFSET_Y = 50
CELL_SIZE = 60
WALL_THICKNESS = 15
CELL_STRIDE = CELL_SIZE + WALL_THICKNESS

# Colors
BG_COLOR = (20, 20, 24)
BOARD_COLOR = (139, 69, 19)
CELL_COLOR = (200, 150, 100)
WALL_COLOR = (200, 50, 50)
HOVER_WALL_COLOR = (100, 150, 100, 150)
HOVER_CELL_COLOR = (255, 255, 150)
P0_COLOR = (255, 255, 255)
P1_COLOR = (0, 0, 0)
TEXT_COLOR = (220, 220, 220)
PANEL_BG = (35, 35, 42)

STATE_STARTUP = 0
STATE_TRAINING = 1
STATE_PLAYING = 2

class PygameGUI:
    def __init__(self, game: QuoridorGame, default_model_path: Optional[str] = None, human_player: int = 0, config: Optional[dict] = None):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Quoridor AI Setup & Battle")
        
        # Font styling
        self.font = pygame.font.SysFont('Arial', 18)
        self.medium_font = pygame.font.SysFont('Arial', 22, bold=True)
        self.large_font = pygame.font.SysFont('Arial', 32, bold=True)
        self.title_font = pygame.font.SysFont('Arial', 40, bold=True)
        
        self.game = game
        self.config = config or {}
        self.human_player = human_player
        self.play_against_idx = 0  # 0 for Model 1, 1 for Model 2
        
        self.bot_thinking = False
        self.bot_thread = None
        self.bot_move: Optional[Move] = None
        self.running = True
        
        # Resign button rect
        self.resign_rect = pygame.Rect(WINDOW_WIDTH - 250, WINDOW_HEIGHT - 80, 200, 50)
        
        # Set up named model choices
        self.model_options = [
            {"name": "AlphaZero (Random Net)", "type": "alphazero", "path": None},
            {"name": "MCTS (Uniform Policy)", "type": "mcts_uniform", "path": None},
            {"name": "Minimax (Depth 8)", "type": "minimax", "path": None}
        ]
        
        checkpoint_dir = "data/checkpoints"
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
            # Sort files by modification time so the most recently trained model is last
            files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
            for f in files:
                self.model_options.append({
                    "name": f"AlphaZero ({f})",
                    "type": "alphazero",
                    "path": os.path.join(checkpoint_dir, f)
                })
                    
        # Default to the most recently modified checkpoint if it exists, otherwise Random Net
        self.model1_idx = len(self.model_options) - 1 if len(self.model_options) > 3 else 0
        
        # Match default model path if passed
        if default_model_path:
            base = os.path.basename(default_model_path)
            found = False
            for idx, opt in enumerate(self.model_options):
                if opt["path"] and os.path.basename(opt["path"]) == base:
                    self.model1_idx = idx
                    found = True
                    break
            if not found and os.path.exists(default_model_path):
                self.model_options.append({
                    "name": f"AlphaZero ({base})",
                    "type": "alphazero",
                    "path": default_model_path
                })
                self.model1_idx = len(self.model_options) - 1
                
        self.model2_idx = 2  # Default Model 2 to Minimax (Depth 8)
        
        # Scrolling
        self.model1_scroll = 0
        self.model2_scroll = 0
        
        # Training state variables
        self.state = STATE_STARTUP
        self.num_training_games_str = "10"
        self.active_input = False
        self.visualize_training_val = True
        self.training_status = ""
        self.training_thread = None
        self.training_done = False
        
        # Loaded model references
        self.model1 = None
        self.model2 = None
        self.model = None
        self.mcts = None
        self.battle_agent_type = "alphazero"
        self.spectator_mcts_p0 = None
        self.spectator_mcts_p1 = None
        
        # History Graph Tracker
        self.history_tracker = HistoryTracker()
        self.current_game_path = []
        
    def _run_mcts(self):
        state_copy = self.game.board.clone()
        curr_p = state_copy.current_player
        
        # Check endgame solver first
        solver = get_solver()
        if solver.should_activate(state_copy):
            _, move = solver.solve(state_copy, curr_p)
            if move is not None:
                self.bot_move = move
                self.bot_thinking = False
                return
        
        # Determine agent type and model
        if self.human_player in (0, 1):
            agent_type = self.battle_agent_type
            mcts_agent = self.mcts
            model = self.model
        else:
            # Spectator mode: Model 1 as P0, Model 2 as P1
            if curr_p == 0:
                agent_type = self.model_options[self.model1_idx]["type"]
                mcts_agent = self.spectator_mcts_p0
            else:
                agent_type = self.model_options[self.model2_idx]["type"]
                mcts_agent = self.spectator_mcts_p1
            
        if agent_type == "minimax":
            import math
            from search.minimax import alphabeta
            from engine.rules import get_all_legal_moves
            _, move = alphabeta(state_copy, depth=8, alpha=-math.inf, beta=math.inf, maximizing_player=True, root_player=curr_p)
            if move is None:
                import random
                move = random.choice(get_all_legal_moves(state_copy))
            self.bot_move = move
        else:
            action_probs = mcts_agent.search(state_copy)
            from models.board_encoder import decode_action
            best_action_idx = max(action_probs.items(), key=lambda x: x[1])[0]
            self.bot_move = decode_action(state_copy, best_action_idx)
            
        self.bot_thinking = False

    def trigger_bot_move(self):
        if not self.bot_thinking and not self.game.is_terminal:
            self.bot_thinking = True
            self.bot_thread = threading.Thread(target=self._run_mcts)
            self.bot_thread.start()

    def _load_agent_by_idx(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        opt = self.model_options[idx]
        
        if opt["type"] == "alphazero":
            model = QuoridorNet().to(device)
            path = opt["path"]
            if path and os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Error loading checkpoint {path}: {e}")
            return model
        elif opt["type"] == "mcts_uniform":
            return UniformModel().to(device)
        else: # minimax
            return UniformModel().to(device)

    def _run_training(self, num_games):
        if num_games <= 0:
            self.training_done = True
            return
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the selected models
        self.model1 = self._load_agent_by_idx(self.model1_idx)
        self.model2 = self._load_agent_by_idx(self.model2_idx)
        
        optimizer1 = optim.Adam(self.model1.parameters(), lr=0.001)
        optimizer2 = optim.Adam(self.model2.parameters(), lr=0.001)
        
        buffer1 = ExperienceBuffer(capacity=10000)
        buffer2 = ExperienceBuffer(capacity=10000)
        batch_size = 32
        
        opt1 = self.model_options[self.model1_idx]
        opt2 = self.model_options[self.model2_idx]
        
        for g in range(1, num_games + 1):
            model1_is_player_0 = (g % 2 == 0)
            self.training_status = f"Self-play game {g}/{num_games} (M1 as P{0 if model1_is_player_0 else 1})..."
            
            from training.self_play_worker import play_co_play_game
            m1_data, m2_data, transitions, winner = play_co_play_game(
                opt1["type"], self.model1, opt2["type"], self.model2, mcts_simulations=50, model1_is_player_0=model1_is_player_0
            )
            
            # Record transitions to history
            path = []
            for board_before, move, board_after in transitions:
                self.history_tracker.record_game_step(board_before, move, board_after)
                path.append(self.history_tracker.get_state_key(board_before))
            
            # Record final outcome
            if winner is not None and len(transitions) > 0:
                path.append(self.history_tracker.get_state_key(transitions[-1][2]))
                self.history_tracker.record_game_outcome(path, winner)
            
            # Add to buffers
            for state, policy, value in m1_data:
                buffer1.add(state, policy, value)
            for state, policy, value in m2_data:
                buffer2.add(state, policy, value)
                
            # Train model 1 (only if alphazero type)
            if opt1["type"] == "alphazero" and len(buffer1) >= batch_size:
                self.training_status = f"Optimizing Model 1 (Game {g}/{num_games})..."
                batch = buffer1.sample(batch_size)
                train_step(self.model1, optimizer1, batch, device=device)
                
            # Train model 2 (only if alphazero type)
            if opt2["type"] == "alphazero" and len(buffer2) >= batch_size:
                self.training_status = f"Optimizing Model 2 (Game {g}/{num_games})..."
                batch = buffer2.sample(batch_size)
                train_step(self.model2, optimizer2, batch, device=device)
                
        # Save checkpoints and history
        self.training_status = "Saving trained models to disk..."
        self.save_trained_models()
        self.history_tracker.save()
        
        self.training_status = "Co-training complete!"
        self.training_done = True

    def save_trained_models(self):
        os.makedirs("data/checkpoints", exist_ok=True)
        opt1 = self.model_options[self.model1_idx]
        opt2 = self.model_options[self.model2_idx]
        
        if opt1["type"] == "alphazero":
            name1 = opt1["name"].replace("AlphaZero (", "").replace(")", "").replace(".pt", "").replace(".pth", "")
            if opt1["path"] is None:
                name1 = "random_model1"
            save_path1 = f"data/checkpoints/{name1}_trained.pt"
            torch.save({'model_state_dict': self.model1.state_dict()}, save_path1)
            
        if opt2["type"] == "alphazero":
            name2 = opt2["name"].replace("AlphaZero (", "").replace(")", "").replace(".pt", "").replace(".pth", "")
            if opt2["path"] is None:
                name2 = "random_model2"
            save_path2 = f"data/checkpoints/{name2}_trained.pt"
            torch.save({'model_state_dict': self.model2.state_dict()}, save_path2)

    def init_next_training_game(self):
        self.train_game = QuoridorGame()
        self.train_game_history = []
        self.train_transitions = []
        self.train_model1_is_player_0 = (self.train_game_idx % 2 == 0)
        
        m1_name = self.model_options[self.model1_idx]["name"]
        m2_name = self.model_options[self.model2_idx]["name"]
        self.training_status = f"Playing game {self.train_game_idx}/{self.num_training_games}..."
        
        if not hasattr(self, 'train_buffer1'):
            self.train_buffer1 = ExperienceBuffer(capacity=10000)
            self.train_buffer2 = ExperienceBuffer(capacity=10000)
            self.train_optimizer1 = optim.Adam(self.model1.parameters(), lr=0.001)
            self.train_optimizer2 = optim.Adam(self.model2.parameters(), lr=0.001)

    def update_visual_training(self):
        if self.train_game.is_terminal:
            winner = self.train_game.winner
            
            # Record transitions to history
            path = []
            for board_before, move, board_after in self.train_transitions:
                self.history_tracker.record_game_step(board_before, move, board_after)
                path.append(self.history_tracker.get_state_key(board_before))
            
            if winner is not None and len(self.train_transitions) > 0:
                path.append(self.history_tracker.get_state_key(self.train_transitions[-1][2]))
                self.history_tracker.record_game_outcome(path, winner)
            
            # Add to buffers
            for state, policy, p in self.train_game_history:
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if p == winner else -1.0
                
                is_model1 = (p == 0 and self.train_model1_is_player_0) or (p == 1 and not self.train_model1_is_player_0)
                if is_model1:
                    self.train_buffer1.add(state, policy, value)
                else:
                    self.train_buffer2.add(state, policy, value)
            
            # Train model parameters
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_size = 32
            opt1 = self.model_options[self.model1_idx]
            opt2 = self.model_options[self.model2_idx]
            
            if opt1["type"] == "alphazero" and len(self.train_buffer1) >= batch_size:
                self.training_status = "Optimizing Model 1..."
                batch = self.train_buffer1.sample(batch_size)
                train_step(self.model1, self.train_optimizer1, batch, device=device)
                
            if opt2["type"] == "alphazero" and len(self.train_buffer2) >= batch_size:
                self.training_status = "Optimizing Model 2..."
                batch = self.train_buffer2.sample(batch_size)
                train_step(self.model2, self.train_optimizer2, batch, device=device)
                
            self.train_game_idx += 1
            if self.train_game_idx > self.num_training_games:
                # Training complete
                self.training_status = "Saving trained models..."
                self.save_trained_models()
                self.history_tracker.save()
                
                # Setup battle agent
                selected_model = self.model1 if self.play_against_idx == 0 else self.model2
                selected_model.eval()
                self.model = selected_model
                
                selected_type = self.model_options[self.model1_idx if self.play_against_idx == 0 else self.model2_idx]["type"]
                self.battle_agent_type = selected_type
                if selected_type != "minimax":
                    self.mcts = MCTS(self.model, num_simulations=400)
                # Init spectator MCTS instances
                if self.human_player not in (0, 1):
                    self.spectator_mcts_p0 = MCTS(self.model1, num_simulations=400) if self.model_options[self.model1_idx]["type"] != "minimax" else None
                    self.spectator_mcts_p1 = MCTS(self.model2, num_simulations=400) if self.model_options[self.model2_idx]["type"] != "minimax" else None
                    
                self.state = STATE_PLAYING
            else:
                self.init_next_training_game()
        else:
            # Run one move of visual game
            curr_p = self.train_game.board.current_player
            if curr_p == 0:
                agent_type = self.model_options[self.model1_idx]["type"]
                model = self.model1
            else:
                agent_type = self.model_options[self.model2_idx]["type"]
                model = self.model2
                
            board_before = self.train_game.board.clone()
            
            if agent_type == "minimax":
                import math
                from search.minimax import alphabeta
                from engine.rules import get_all_legal_moves
                _, move = alphabeta(self.train_game.board, depth=8, alpha=-math.inf, beta=math.inf, maximizing_player=True, root_player=curr_p)
                if move is None:
                    import random
                    move = random.choice(get_all_legal_moves(self.train_game.board))
                policy = np.zeros(136, dtype=np.float32)
                from models.board_encoder import encode_action
                try:
                    action_idx = encode_action(board_before, move)
                    policy[action_idx] = 1.0
                except:
                    pass
            else:
                # alphazero or mcts_uniform — reuse persistent MCTS
                if curr_p == 0:
                    if not hasattr(self, '_train_mcts_p0') or self._train_mcts_p0 is None:
                        self._train_mcts_p0 = MCTS(model, num_simulations=50)
                    mcts = self._train_mcts_p0
                else:
                    if not hasattr(self, '_train_mcts_p1') or self._train_mcts_p1 is None:
                        self._train_mcts_p1 = MCTS(model, num_simulations=50)
                    mcts = self._train_mcts_p1
                action_probs = mcts.search(self.train_game.board)
                policy = np.zeros(136, dtype=np.float32)
                for a, p in action_probs.items():
                    policy[a] = p
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                chosen_action = np.random.choice(actions, p=probs)
                move = decode_action(self.train_game.board, chosen_action)
                
            self.train_game_history.append((encode_board(board_before), policy, curr_p))
            self.train_game.apply_move(move)
            board_after = self.train_game.board.clone()
            self.train_transitions.append((board_before, move, board_after))
            
            # Short delay so the user can easily observe the moves
            pygame.time.delay(200)

    def handle_click(self, pos):
        if self.state != STATE_PLAYING:
            return
            
        if self.bot_thinking or self.game.is_terminal:
            return
            
        if self.resign_rect.collidepoint(pos):
            self.game.is_terminal = True
            self.game.winner = 1 - self.human_player
            return
            
        if self.game.board.current_player != self.human_player:
            return
            
        x, y = pos
        rel_x = x - BOARD_OFFSET_X
        rel_y = y - BOARD_OFFSET_Y
        
        if rel_x < 0 or rel_y < 0:
            return
            
        col = rel_x // CELL_STRIDE
        row = rel_y // CELL_STRIDE
        
        in_cell_x = (rel_x % CELL_STRIDE) < CELL_SIZE
        in_cell_y = (rel_y % CELL_STRIDE) < CELL_SIZE
        
        legal_moves = self.game.get_legal_moves()
        move = None
        
        if in_cell_x and in_cell_y:
            if 0 <= row < 9 and 0 <= col < 9:
                move = PawnMove(row, col)
        elif in_cell_x and not in_cell_y:
            if 0 <= row < 8 and 0 <= col < 8:
                move = WallMove(row, col, 'h')
        elif not in_cell_x and in_cell_y:
            if 0 <= row < 8 and 0 <= col < 8:
                move = WallMove(row, col, 'v')
            
        if move in legal_moves:
            board_before = self.game.board.clone()
            self.game.apply_move(move)
            board_after = self.game.board.clone()
            
            self.history_tracker.record_game_step(board_before, move, board_after)
            self.current_game_path.append(self.history_tracker.get_state_key(board_before))
            
    def draw_startup(self):
        self.screen.fill((20, 20, 24))
        
        # Title
        title_surf = self.title_font.render("QUORIDOR AI SETUP", True, (255, 255, 255))
        self.screen.blit(title_surf, (WINDOW_WIDTH // 2 - title_surf.get_width() // 2, 30))
        
        # Model 1 Panel (P0)
        panel1_rect = pygame.Rect(50, 90, 480, 320)
        pygame.draw.rect(self.screen, (32, 32, 38), panel1_rect, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 70), panel1_rect, 2, border_radius=8)
        lbl1 = self.medium_font.render("Model 1 (Player 0 / Red)", True, (255, 100, 100))
        self.screen.blit(lbl1, (70, 105))
        
        # Model 2 Panel (P1)
        panel2_rect = pygame.Rect(570, 90, 480, 320)
        pygame.draw.rect(self.screen, (32, 32, 38), panel2_rect, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 70), panel2_rect, 2, border_radius=8)
        lbl2 = self.medium_font.render("Model 2 (Player 1 / Blue)", True, (100, 150, 255))
        self.screen.blit(lbl2, (590, 105))
        
        # Render items for Model 1
        y_start = 145
        for i in range(self.model1_scroll, min(self.model1_scroll + 4, len(self.model_options))):
            idx = i - self.model1_scroll
            item_rect = pygame.Rect(70, y_start + idx * 45, 440, 36)
            is_selected = (i == self.model1_idx)
            
            mouse_pos = pygame.mouse.get_pos()
            is_hover = item_rect.collidepoint(mouse_pos)
            
            bg = (0, 100, 200) if is_selected else ((48, 48, 56) if is_hover else (40, 40, 46))
            pygame.draw.rect(self.screen, bg, item_rect, border_radius=6)
            
            color = (255, 255, 255) if (is_selected or is_hover) else (180, 180, 180)
            txt = self.font.render(self.model_options[i]["name"], True, color)
            self.screen.blit(txt, (item_rect.x + 10, item_rect.y + 8))
            
        # Draw Scroll Arrows if needed for Model 1
        if len(self.model_options) > 4:
            up_rect1 = pygame.Rect(485, 145, 30, 30)
            pygame.draw.rect(self.screen, (40, 40, 46), up_rect1, border_radius=4)
            down_rect1 = pygame.Rect(485, 280, 30, 30)
            pygame.draw.rect(self.screen, (40, 40, 46), down_rect1, border_radius=4)
            
            pygame.draw.polygon(self.screen, (200, 200, 200), [(492, 165), (500, 153), (508, 165)])
            pygame.draw.polygon(self.screen, (200, 200, 200), [(492, 290), (500, 302), (508, 290)])
            
        # Render items for Model 2
        for i in range(self.model2_scroll, min(self.model2_scroll + 4, len(self.model_options))):
            idx = i - self.model2_scroll
            item_rect = pygame.Rect(590, y_start + idx * 45, 440, 36)
            is_selected = (i == self.model2_idx)
            
            mouse_pos = pygame.mouse.get_pos()
            is_hover = item_rect.collidepoint(mouse_pos)
            
            bg = (0, 100, 200) if is_selected else ((48, 48, 56) if is_hover else (40, 40, 46))
            pygame.draw.rect(self.screen, bg, item_rect, border_radius=6)
            
            color = (255, 255, 255) if (is_selected or is_hover) else (180, 180, 180)
            txt = self.font.render(self.model_options[i]["name"], True, color)
            self.screen.blit(txt, (item_rect.x + 10, item_rect.y + 8))
            
        # Draw Scroll Arrows if needed for Model 2
        if len(self.model_options) > 4:
            up_rect2 = pygame.Rect(1005, 145, 30, 30)
            pygame.draw.rect(self.screen, (40, 40, 46), up_rect2, border_radius=4)
            down_rect2 = pygame.Rect(1005, 280, 30, 30)
            pygame.draw.rect(self.screen, (40, 40, 46), down_rect2, border_radius=4)
            
            pygame.draw.polygon(self.screen, (200, 200, 200), [(1012, 165), (1020, 153), (1028, 165)])
            pygame.draw.polygon(self.screen, (200, 200, 200), [(1012, 290), (1020, 302), (1028, 290)])
            
        # Bottom panels
        # 1. Training Setup Panel
        train_panel = pygame.Rect(50, 430, 480, 180)
        pygame.draw.rect(self.screen, (32, 32, 38), train_panel, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 70), train_panel, 2, border_radius=8)
        self.screen.blit(self.medium_font.render("Training Configuration", True, (255, 200, 100)), (70, 445))
        
        self.screen.blit(self.font.render("Self-play games to train simultaneously:", True, (180, 180, 180)), (70, 475))
        input_rect = pygame.Rect(70, 505, 200, 36)
        border_col = (0, 150, 255) if self.active_input else (100, 100, 110)
        pygame.draw.rect(self.screen, (40, 40, 46), input_rect, border_radius=6)
        pygame.draw.rect(self.screen, border_col, input_rect, 2, border_radius=6)
        
        input_val_surf = self.medium_font.render(self.num_training_games_str, True, (255, 255, 255))
        self.screen.blit(input_val_surf, (input_rect.x + 10, input_rect.y + 6))
        
        # Visualize Training Checkbox
        vis_box = pygame.Rect(70, 560, 24, 24)
        pygame.draw.rect(self.screen, (40, 40, 46), vis_box, border_radius=4)
        pygame.draw.rect(self.screen, (100, 100, 110) if not self.visualize_training_val else (0, 150, 255), vis_box, 2, border_radius=4)
        if self.visualize_training_val:
            pygame.draw.line(self.screen, (0, 150, 255), (vis_box.x + 5, vis_box.y + 12), (vis_box.x + 10, vis_box.y + 18), 3)
            pygame.draw.line(self.screen, (0, 150, 255), (vis_box.x + 10, vis_box.y + 18), (vis_box.x + 19, vis_box.y + 6), 3)
        self.screen.blit(self.font.render("Visualize Training Matches", True, (200, 200, 200)), (105, 562))
        
        # 2. Match Setup Panel
        match_panel = pygame.Rect(570, 430, 480, 180)
        pygame.draw.rect(self.screen, (32, 32, 38), match_panel, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 70), match_panel, 2, border_radius=8)
        self.screen.blit(self.medium_font.render("Match Configuration", True, (100, 255, 200)), (590, 445))
        
        self.screen.blit(self.font.render("Human Player:", True, (180, 180, 180)), (590, 475))
        
        # Human Player Buttons
        btn_p0 = pygame.Rect(590, 500, 130, 32)
        btn_p1 = pygame.Rect(730, 500, 130, 32)
        btn_spec = pygame.Rect(870, 500, 160, 32)
        
        mouse_pos = pygame.mouse.get_pos()
        
        col_p0 = (0, 150, 255) if self.human_player == 0 else ((48, 48, 56) if btn_p0.collidepoint(mouse_pos) else (40, 40, 46))
        col_p1 = (0, 150, 255) if self.human_player == 1 else ((48, 48, 56) if btn_p1.collidepoint(mouse_pos) else (40, 40, 46))
        col_spec = (0, 150, 255) if (self.human_player not in (0, 1)) else ((48, 48, 56) if btn_spec.collidepoint(mouse_pos) else (40, 40, 46))
        
        pygame.draw.rect(self.screen, col_p0, btn_p0, border_radius=4)
        pygame.draw.rect(self.screen, col_p1, btn_p1, border_radius=4)
        pygame.draw.rect(self.screen, col_spec, btn_spec, border_radius=4)
        
        self.screen.blit(self.font.render("P0 (Go First)", True, (255, 255, 255)), (btn_p0.x + 12, btn_p0.y + 6))
        self.screen.blit(self.font.render("P1 (Go Second)", True, (255, 255, 255)), (btn_p1.x + 12, btn_p1.y + 6))
        self.screen.blit(self.font.render("Spectator", True, (255, 255, 255)), (btn_spec.x + 12, btn_spec.y + 6))
        
        # Play opponent selection
        if self.human_player in (0, 1):
            self.screen.blit(self.font.render("Play Against Opponent:", True, (180, 180, 180)), (590, 542))
            btn_opp1 = pygame.Rect(590, 568, 210, 32)
            btn_opp2 = pygame.Rect(810, 568, 210, 32)
            
            col_opp1 = (0, 150, 255) if self.play_against_idx == 0 else ((48, 48, 56) if btn_opp1.collidepoint(mouse_pos) else (40, 40, 46))
            col_opp2 = (0, 150, 255) if self.play_against_idx == 1 else ((48, 48, 56) if btn_opp2.collidepoint(mouse_pos) else (40, 40, 46))
            
            pygame.draw.rect(self.screen, col_opp1, btn_opp1, border_radius=4)
            pygame.draw.rect(self.screen, col_opp2, btn_opp2, border_radius=4)
            
            self.screen.blit(self.font.render("Model 1", True, (255, 255, 255)), (btn_opp1.x + 15, btn_opp1.y + 6))
            self.screen.blit(self.font.render("Model 2", True, (255, 255, 255)), (btn_opp2.x + 15, btn_opp2.y + 6))
            
        # Start button
        start_btn = pygame.Rect(230, 640, 300, 50)
        is_hover_start = start_btn.collidepoint(mouse_pos)
        start_col = (46, 204, 113) if is_hover_start else (39, 174, 96)
        pygame.draw.rect(self.screen, start_col, start_btn, border_radius=8)
        
        start_txt = self.medium_font.render("START MATCH", True, (255, 255, 255))
        self.screen.blit(start_txt, (start_btn.centerx - start_txt.get_width() // 2, start_btn.centery - start_txt.get_height() // 2))

        # Quit button
        quit_btn = pygame.Rect(570, 640, 300, 50)
        is_hover_quit = quit_btn.collidepoint(mouse_pos)
        quit_col = (231, 76, 60) if is_hover_quit else (192, 57, 43)
        pygame.draw.rect(self.screen, quit_col, quit_btn, border_radius=8)
        
        quit_txt = self.medium_font.render("QUIT GAME", True, (255, 255, 255))
        self.screen.blit(quit_txt, (quit_btn.centerx - quit_txt.get_width() // 2, quit_btn.centery - quit_txt.get_height() // 2))

    def draw_training(self):
        self.screen.fill(BG_COLOR)
        title = self.large_font.render("Co-Training in progress...", True, TEXT_COLOR)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 250))
        
        status = self.font.render(self.training_status, True, (255, 200, 100))
        self.screen.blit(status, (WINDOW_WIDTH // 2 - status.get_width() // 2, 330))

        # Quit / Abort button
        mouse_pos = pygame.mouse.get_pos()
        quit_btn = pygame.Rect(WINDOW_WIDTH // 2 - 150, 480, 300, 50)
        is_hover_quit = quit_btn.collidepoint(mouse_pos)
        quit_col = (231, 76, 60) if is_hover_quit else (192, 57, 43)
        pygame.draw.rect(self.screen, quit_col, quit_btn, border_radius=8)
        
        quit_txt = self.medium_font.render("ABORT & QUIT", True, (255, 255, 255))
        self.screen.blit(quit_txt, (quit_btn.centerx - quit_txt.get_width() // 2, quit_btn.centery - quit_txt.get_height() // 2))

    def draw_visual_training(self):
        self.screen.fill((20, 20, 25))
        
        # Board Draw
        board_rect = pygame.Rect(
            BOARD_OFFSET_X - 5, BOARD_OFFSET_Y - 5,
            9 * CELL_STRIDE - WALL_THICKNESS + 10,
            9 * CELL_STRIDE - WALL_THICKNESS + 10
        )
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect)
        
        for r in range(9):
            for c in range(9):
                cx = BOARD_OFFSET_X + c * CELL_STRIDE
                cy = BOARD_OFFSET_Y + r * CELL_STRIDE
                rect = pygame.Rect(cx, cy, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, CELL_COLOR, rect)
                
                if self.train_game.board.pawn_positions[0] == (r, c):
                    pygame.draw.circle(self.screen, P0_COLOR, rect.center, CELL_SIZE // 2 - 5)
                    pygame.draw.circle(self.screen, (255, 100, 100), rect.center, 5)
                elif self.train_game.board.pawn_positions[1] == (r, c):
                    pygame.draw.circle(self.screen, P1_COLOR, rect.center, CELL_SIZE // 2 - 5)
                    pygame.draw.circle(self.screen, (100, 150, 255), rect.center, 5)

        for hr, hc in self.train_game.board.h_walls:
            x = BOARD_OFFSET_X + hc * CELL_STRIDE
            y = BOARD_OFFSET_Y + hr * CELL_STRIDE + CELL_SIZE
            w = CELL_STRIDE + CELL_SIZE
            h = WALL_THICKNESS
            pygame.draw.rect(self.screen, WALL_COLOR, pygame.Rect(x, y, w, h))
            
        for vr, vc in self.train_game.board.v_walls:
            x = BOARD_OFFSET_X + vc * CELL_STRIDE + CELL_SIZE
            y = BOARD_OFFSET_Y + vr * CELL_STRIDE
            w = WALL_THICKNESS
            h = CELL_STRIDE + CELL_SIZE
            pygame.draw.rect(self.screen, WALL_COLOR, pygame.Rect(x, y, w, h))

        # Panel
        panel_x = BOARD_OFFSET_X + 9 * CELL_STRIDE + 20
        panel_w = WINDOW_WIDTH - panel_x
        pygame.draw.rect(self.screen, PANEL_BG, pygame.Rect(panel_x, 0, panel_w, WINDOW_HEIGHT))
        
        title_surf = self.medium_font.render("LIVE TRAINING BATTLE", True, (255, 200, 100))
        self.screen.blit(title_surf, (panel_x + 20, 20))
        
        status_surf = self.font.render(self.training_status, True, TEXT_COLOR)
        self.screen.blit(status_surf, (panel_x + 20, 60))
        
        m1_name = self.model_options[self.model1_idx]["name"]
        m2_name = self.model_options[self.model2_idx]["name"]
        self.screen.blit(self.font.render(f"Model 1 (P0): {m1_name}", True, (255, 150, 150)), (panel_x + 20, 110))
        self.screen.blit(self.font.render(f"Model 2 (P1): {m2_name}", True, (150, 200, 255)), (panel_x + 20, 140))
        
        self.screen.blit(self.medium_font.render(f"Game {self.train_game_idx} of {self.num_training_games}", True, (255, 255, 255)), (panel_x + 20, 190))
        
        history_title = self.font.render("Game Move History:", True, TEXT_COLOR)
        self.screen.blit(history_title, (panel_x + 20, 240))
        
        hist = self.train_game.board.move_history
        show_hist = hist[-15:]
        start_y = 270
        for i, m in enumerate(show_hist):
            real_i = len(hist) - len(show_hist) + i
            player = real_i % 2
            move_str = f"{real_i+1}. P{player}: {m}"
            m_surf = self.font.render(move_str, True, TEXT_COLOR)
            self.screen.blit(m_surf, (panel_x + 20, start_y + i * 25))
            
        mouse_pos = pygame.mouse.get_pos()
        quit_btn = pygame.Rect(panel_x + 20, WINDOW_HEIGHT - 80, panel_w - 40, 50)
        is_hover_quit = quit_btn.collidepoint(mouse_pos)
        quit_col = (231, 76, 60) if is_hover_quit else (192, 57, 43)
        pygame.draw.rect(self.screen, quit_col, quit_btn, border_radius=8)
        
        quit_txt = self.medium_font.render("ABORT & QUIT", True, (255, 255, 255))
        self.screen.blit(quit_txt, (quit_btn.centerx - quit_txt.get_width() // 2, quit_btn.centery - quit_txt.get_height() // 2))

    def draw_playing(self):
        self.screen.fill((30, 30, 35))
        
        board_rect = pygame.Rect(
            BOARD_OFFSET_X - 5, BOARD_OFFSET_Y - 5,
            9 * CELL_STRIDE - WALL_THICKNESS + 10,
            9 * CELL_STRIDE - WALL_THICKNESS + 10
        )
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect)
        
        mouse_pos = pygame.mouse.get_pos()
        legal_moves = [] if self.game.is_terminal else self.game.get_legal_moves()
        
        hover_move = None
        if self.game.board.current_player == self.human_player and not self.bot_thinking and not self.game.is_terminal:
            rel_x = mouse_pos[0] - BOARD_OFFSET_X
            rel_y = mouse_pos[1] - BOARD_OFFSET_Y
            if rel_x >= 0 and rel_y >= 0:
                col = rel_x // CELL_STRIDE
                row = rel_y // CELL_STRIDE
                in_cell_x = (rel_x % CELL_STRIDE) < CELL_SIZE
                in_cell_y = (rel_y % CELL_STRIDE) < CELL_SIZE
                
                if in_cell_x and in_cell_y and 0 <= row < 9 and 0 <= col < 9:
                    hover_move = PawnMove(row, col)
                elif in_cell_x and not in_cell_y and 0 <= row < 8 and 0 <= col < 8:
                    hover_move = WallMove(row, col, 'h')
                elif not in_cell_x and in_cell_y and 0 <= row < 8 and 0 <= col < 8:
                    hover_move = WallMove(row, col, 'v')
                    
                if hover_move not in legal_moves:
                    hover_move = None

        for r in range(9):
            for c in range(9):
                cx = BOARD_OFFSET_X + c * CELL_STRIDE
                cy = BOARD_OFFSET_Y + r * CELL_STRIDE
                rect = pygame.Rect(cx, cy, CELL_SIZE, CELL_SIZE)
                
                color = CELL_COLOR
                if hover_move and isinstance(hover_move, PawnMove) and hover_move.to_row == r and hover_move.to_col == c:
                    color = HOVER_CELL_COLOR
                    
                pygame.draw.rect(self.screen, color, rect)
                
                if self.game.board.pawn_positions[0] == (r, c):
                    pygame.draw.circle(self.screen, P0_COLOR, rect.center, CELL_SIZE // 2 - 5)
                    pygame.draw.circle(self.screen, (255, 100, 100), rect.center, 5)
                elif self.game.board.pawn_positions[1] == (r, c):
                    pygame.draw.circle(self.screen, P1_COLOR, rect.center, CELL_SIZE // 2 - 5)
                    pygame.draw.circle(self.screen, (100, 150, 255), rect.center, 5)

        for hr, hc in self.game.board.h_walls:
            x = BOARD_OFFSET_X + hc * CELL_STRIDE
            y = BOARD_OFFSET_Y + hr * CELL_STRIDE + CELL_SIZE
            w = CELL_STRIDE + CELL_SIZE
            h = WALL_THICKNESS
            pygame.draw.rect(self.screen, WALL_COLOR, pygame.Rect(x, y, w, h))
            
        for vr, vc in self.game.board.v_walls:
            x = BOARD_OFFSET_X + vc * CELL_STRIDE + CELL_SIZE
            y = BOARD_OFFSET_Y + vr * CELL_STRIDE
            w = WALL_THICKNESS
            h = CELL_STRIDE + CELL_SIZE
            pygame.draw.rect(self.screen, WALL_COLOR, pygame.Rect(x, y, w, h))

        if hover_move and isinstance(hover_move, WallMove):
            s = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            if hover_move.orientation == 'h':
                x = BOARD_OFFSET_X + hover_move.col * CELL_STRIDE
                y = BOARD_OFFSET_Y + hover_move.row * CELL_STRIDE + CELL_SIZE
                w = CELL_STRIDE + CELL_SIZE
                h = WALL_THICKNESS
                pygame.draw.rect(s, HOVER_WALL_COLOR, pygame.Rect(x, y, w, h))
            else:
                x = BOARD_OFFSET_X + hover_move.col * CELL_STRIDE + CELL_SIZE
                y = BOARD_OFFSET_Y + hover_move.row * CELL_STRIDE
                w = WALL_THICKNESS
                h = CELL_STRIDE + CELL_SIZE
                pygame.draw.rect(s, HOVER_WALL_COLOR, pygame.Rect(x, y, w, h))
            self.screen.blit(s, (0, 0))

        panel_x = BOARD_OFFSET_X + 9 * CELL_STRIDE + 20
        panel_w = WINDOW_WIDTH - panel_x
        pygame.draw.rect(self.screen, PANEL_BG, pygame.Rect(panel_x, 0, panel_w, WINDOW_HEIGHT))
        
        current_p = self.game.board.current_player
        
        turn_text = f"Turn: Player {current_p} {'(You)' if current_p == self.human_player else '(Bot)'}"
        if self.human_player not in (0, 1):
            model_lbl = "Model 1" if current_p == 0 else "Model 2"
            turn_text = f"Turn: Player {current_p} ({model_lbl})"
            
        if self.game.is_terminal:
            turn_text = f"Game Over! Player {self.game.winner} wins!"
        elif self.bot_thinking:
            turn_text = "Bot is thinking..."
            
        text_surf = self.large_font.render(turn_text, True, TEXT_COLOR)
        self.screen.blit(text_surf, (panel_x + 20, 20))
        
        p0_walls = self.game.board.walls_remaining[0]
        p1_walls = self.game.board.walls_remaining[1]
        w_text = self.font.render(f"P0 Walls: {p0_walls}  |  P1 Walls: {p1_walls}", True, TEXT_COLOR)
        self.screen.blit(w_text, (panel_x + 20, 70))
        
        # Display MCTS / Minimax lookahead depth
        lookahead_depth = 0
        if current_p != self.human_player and not self.game.is_terminal:
            if self.battle_agent_type == "minimax":
                lookahead_depth = 8
            elif self.mcts is not None:
                lookahead_depth = getattr(self.mcts, 'last_search_depth', 0)
        elif self.human_player not in (0, 1) and not self.game.is_terminal:
            # Spectator mode
            curr_type = self.model_options[self.model1_idx if current_p == 0 else self.model2_idx]["type"]
            if curr_type == "minimax":
                lookahead_depth = 8
            elif hasattr(self, 'model1') and hasattr(self, 'model2'):
                # We can estimate depth from MCTS node search
                lookahead_depth = 8 # baseline
                
        if lookahead_depth > 0:
            depth_text = self.medium_font.render(f"AI Predict Depth: {lookahead_depth} moves", True, (0, 255, 150))
            self.screen.blit(depth_text, (panel_x + 20, 105))
            
        history_title = self.font.render("Move History:", True, TEXT_COLOR)
        self.screen.blit(history_title, (panel_x + 20, 140))
        
        hist = self.game.board.move_history
        show_hist = hist[-20:]
        start_y = 170
        for i, m in enumerate(show_hist):
            real_i = len(hist) - len(show_hist) + i
            player = real_i % 2
            move_str = f"{real_i+1}. P{player}: {m}"
            m_surf = self.font.render(move_str, True, TEXT_COLOR)
            self.screen.blit(m_surf, (panel_x + 20, start_y + i * 25))
            
        if not self.game.is_terminal and self.human_player in (0, 1):
            color = (200, 50, 50) if self.resign_rect.collidepoint(mouse_pos) else (150, 50, 50)
            pygame.draw.rect(self.screen, color, self.resign_rect)
            res_surf = self.large_font.render("Resign", True, TEXT_COLOR)
            res_rect = res_surf.get_rect(center=self.resign_rect.center)
            self.screen.blit(res_surf, res_rect)

        # Draw Quit / Quit Game button
        quit_rect = pygame.Rect(WINDOW_WIDTH - 250, WINDOW_HEIGHT - 145, 200, 50) if not self.game.is_terminal else pygame.Rect(WINDOW_WIDTH - 250, WINDOW_HEIGHT - 80, 200, 50)
        color_q = (231, 76, 60) if quit_rect.collidepoint(mouse_pos) else (192, 57, 43)
        pygame.draw.rect(self.screen, color_q, quit_rect, border_radius=6)
        quit_surf = self.large_font.render("Quit Game", True, TEXT_COLOR)
        quit_rect_text = quit_surf.get_rect(center=quit_rect.center)
        self.screen.blit(quit_surf, quit_rect_text)

    def draw(self):
        if self.state == STATE_STARTUP:
            self.draw_startup()
        elif self.state == STATE_TRAINING:
            if self.visual_training:
                self.draw_visual_training()
            else:
                self.draw_training()
        elif self.state == STATE_PLAYING:
            self.draw_playing()
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        
                if self.state == STATE_STARTUP:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        x, y = event.pos
                        y_start = 145
                        
                        # Model 1 list items clicks
                        for i in range(self.model1_scroll, min(self.model1_scroll + 4, len(self.model_options))):
                            idx = i - self.model1_scroll
                            item_rect = pygame.Rect(70, y_start + idx * 45, 440, 36)
                            if item_rect.collidepoint(event.pos):
                                self.model1_idx = i
                                
                        # Model 2 list items clicks
                        for i in range(self.model2_scroll, min(self.model2_scroll + 4, len(self.model_options))):
                            idx = i - self.model2_scroll
                            item_rect = pygame.Rect(590, y_start + idx * 45, 440, 36)
                            if item_rect.collidepoint(event.pos):
                                self.model2_idx = i
                                
                        # Scroll arrows
                        if len(self.model_options) > 4:
                            if pygame.Rect(485, 145, 30, 30).collidepoint(event.pos):
                                self.model1_scroll = max(0, self.model1_scroll - 1)
                            elif pygame.Rect(485, 280, 30, 30).collidepoint(event.pos):
                                self.model1_scroll = min(len(self.model_options) - 4, self.model1_scroll + 1)
                                
                            if pygame.Rect(1005, 145, 30, 30).collidepoint(event.pos):
                                self.model2_scroll = max(0, self.model2_scroll - 1)
                            elif pygame.Rect(1005, 280, 30, 30).collidepoint(event.pos):
                                self.model2_scroll = min(len(self.model_options) - 4, self.model2_scroll + 1)
                                
                        # Training games input rect click
                        input_rect = pygame.Rect(70, 505, 200, 36)
                        if input_rect.collidepoint(event.pos):
                            self.active_input = True
                        else:
                            self.active_input = False
                            
                        # Visualize Checkbox Click
                        vis_box = pygame.Rect(70, 560, 24, 24)
                        if vis_box.collidepoint(event.pos):
                            self.visualize_training_val = not self.visualize_training_val
                            
                        # Human player buttons click
                        btn_p0 = pygame.Rect(590, 500, 130, 32)
                        btn_p1 = pygame.Rect(730, 500, 130, 32)
                        btn_spec = pygame.Rect(870, 500, 160, 32)
                        
                        if btn_p0.collidepoint(event.pos):
                            self.human_player = 0
                        elif btn_p1.collidepoint(event.pos):
                            self.human_player = 1
                        elif btn_spec.collidepoint(event.pos):
                            self.human_player = None
                            
                        # Opponent choice clicks
                        if self.human_player in (0, 1):
                            btn_opp1 = pygame.Rect(590, 568, 210, 32)
                            btn_opp2 = pygame.Rect(810, 568, 210, 32)
                            if btn_opp1.collidepoint(event.pos):
                                self.play_against_idx = 0
                            elif btn_opp2.collidepoint(event.pos):
                                self.play_against_idx = 1
                                
                        # Start button click
                        start_btn = pygame.Rect(230, 640, 300, 50)
                        if start_btn.collidepoint(event.pos):
                            try:
                                num_games = int(self.num_training_games_str)
                            except:
                                num_games = 0
                                
                            self.start_training(num_games)
                                
                        # Quit button click
                        quit_btn = pygame.Rect(570, 640, 300, 50)
                        if quit_btn.collidepoint(event.pos):
                            self.running = False

                    elif event.type == pygame.KEYDOWN:
                        if self.active_input:
                            if event.key == pygame.K_RETURN:
                                self.active_input = False
                            elif event.key == pygame.K_BACKSPACE:
                                self.num_training_games_str = self.num_training_games_str[:-1]
                            elif event.unicode.isdigit():
                                self.num_training_games_str += event.unicode
                                
                elif self.state == STATE_TRAINING:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        panel_x = BOARD_OFFSET_X + 9 * CELL_STRIDE + 20
                        panel_w = WINDOW_WIDTH - panel_x
                        if self.visual_training:
                            quit_btn = pygame.Rect(panel_x + 20, WINDOW_HEIGHT - 80, panel_w - 40, 50)
                        else:
                            quit_btn = pygame.Rect(WINDOW_WIDTH // 2 - 150, 480, 300, 50)
                        if quit_btn.collidepoint(event.pos):
                            self.running = False
                            
                elif self.state == STATE_PLAYING:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.handle_click(event.pos)
                        # Check Quit button click
                        quit_rect = pygame.Rect(WINDOW_WIDTH - 250, WINDOW_HEIGHT - 145, 200, 50) if not self.game.is_terminal else pygame.Rect(WINDOW_WIDTH - 250, WINDOW_HEIGHT - 80, 200, 50)
                        if quit_rect.collidepoint(event.pos):
                            self.running = False
                            
            if self.state == STATE_TRAINING:
                if self.visual_training:
                    self.update_visual_training()
                else:
                    if self.training_done:
                        selected_model = self.model1 if self.play_against_idx == 0 else self.model2
                        selected_model.eval()
                        self.model = selected_model
                        selected_type = self.model_options[self.model1_idx if self.play_against_idx == 0 else self.model2_idx]["type"]
                        self.battle_agent_type = selected_type
                        if selected_type != "minimax":
                            self.mcts = MCTS(self.model, num_simulations=400)
                        if self.human_player not in (0, 1):
                            self.spectator_mcts_p0 = MCTS(self.model1, num_simulations=400) if self.model_options[self.model1_idx]["type"] != "minimax" else None
                            self.spectator_mcts_p1 = MCTS(self.model2, num_simulations=400) if self.model_options[self.model2_idx]["type"] != "minimax" else None
                        self.state = STATE_PLAYING
                    
            if self.state == STATE_PLAYING:
                if self.bot_move is not None:
                    board_before = self.game.board.clone()
                    self.game.apply_move(self.bot_move)
                    board_after = self.game.board.clone()
                    
                    self.history_tracker.record_game_step(board_before, self.bot_move, board_after)
                    self.current_game_path.append(self.history_tracker.get_state_key(board_before))
                    self.bot_move = None
                    
                if not self.game.is_terminal:
                    is_bot_turn = False
                    if self.human_player not in (0, 1):
                        is_bot_turn = True
                    elif self.game.board.current_player != self.human_player:
                        is_bot_turn = True
                        
                    if is_bot_turn and not self.bot_thinking:
                        self.trigger_bot_move()
                else:
                    if len(self.current_game_path) > 0:
                        final_key = self.history_tracker.get_state_key(self.game.board)
                        self.current_game_path.append(final_key)
                        self.history_tracker.record_game_outcome(self.current_game_path, self.game.winner)
                        self.history_tracker.save()
                        self.current_game_path = []
                    
            self.draw()
            clock.tick(30)
            
        if len(self.current_game_path) > 0:
            self.history_tracker.save()
            
        pygame.quit()

    def start_training(self, num_games):
        self.num_training_games = num_games
        self.train_game_idx = 1
        self.visual_training = self.visualize_training_val
        
        # Load the selected agents
        self.model1 = self._load_agent_by_idx(self.model1_idx)
        self.model2 = self._load_agent_by_idx(self.model2_idx)
        
        if num_games > 0:
            if self.visual_training:
                self.state = STATE_TRAINING
                self.init_next_training_game()
            else:
                self.state = STATE_TRAINING
                self.training_done = False
                self.training_thread = threading.Thread(target=self._run_training, args=(num_games,))
                self.training_thread.start()
        else:
            # Start battle directly
            selected_model = self.model1 if self.play_against_idx == 0 else self.model2
            selected_model.eval()
            self.model = selected_model
            
            selected_type = self.model_options[self.model1_idx if self.play_against_idx == 0 else self.model2_idx]["type"]
            self.battle_agent_type = selected_type
            if selected_type != "minimax":
                self.mcts = MCTS(self.model, num_simulations=400)
            # Init spectator MCTS instances
            if self.human_player not in (0, 1):
                self.spectator_mcts_p0 = MCTS(self.model1, num_simulations=400) if self.model_options[self.model1_idx]["type"] != "minimax" else None
                self.spectator_mcts_p1 = MCTS(self.model2, num_simulations=400) if self.model_options[self.model2_idx]["type"] != "minimax" else None
                
            self.state = STATE_PLAYING
