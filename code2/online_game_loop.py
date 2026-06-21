import time
import torch
import numpy as np

from adapters.playwright_adapter import PlaywrightAdapter
from training.experience_buffer import ExperienceBuffer
from training.local_trainer import LocalTrainer
from models.quoridor_net import QuoridorNet
from strategy.strategy_guide import StrategyGuide
from search.transposition_table import MCTSTranspositionTable, SolverTranspositionTable
from search.mcts import MCTS
from game_engine.board import board_to_tensor

def run_online_loop(num_games=1, time_budget_ms=3000):
    """
    Main loop to play games online using Playwright, save experiences, and train locally.
    Enforces the strict 3000ms budget per move to ensure games finish in 3-5 minutes.
    """
    buffer = ExperienceBuffer()
    model = QuoridorNet()
    trainer = LocalTrainer(model, buffer)
    trainer.load_checkpoint()
    
    guide = StrategyGuide()
    mcts = MCTS()
    
    for game_idx in range(num_games):
        adapter = PlaywrightAdapter(headless=True)
        
        try:
            adapter.start()
            
            mcts_tt = MCTSTranspositionTable()
            endgame_tt = SolverTranspositionTable()
            
            game_experiences = []
            
            while not adapter.check_game_over():
                # 1. Parse current board from the online UI
                board = adapter.parse_board_state()
                
                # We assume the bot acts if it's the bot's turn (simplified here)
                # In reality, you'd check if the UI is prompting for a move.
                
                # 2. Get strategy vector and state tensor
                strat_vec = guide.compute_strategy_vector(board)
                tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0)
                
                # 3. Run MCTS within strictly enforced time limit
                moves, probs = mcts.search(board, model, guide, mcts_tt, time_budget_ms=time_budget_ms)
                
                if len(moves) == 0:
                    break # Safety fallback
                
                # 4. Pick best move
                best_idx = np.argmax(probs)
                best_move = moves[best_idx]
                
                # 5. Execute move via Playwright
                adapter.execute_move(best_move)
                
                # 6. Save experience temporarily (outcome unknown)
                policy_target = torch.zeros(1, 136) # Dummy padding for logic
                policy_target[0, best_idx] = 1.0 # One hot for the taken action for simplicity
                
                game_experiences.append({
                    'state': tensor,
                    'strategy': torch.tensor(strat_vec, dtype=torch.float32).unsqueeze(0),
                    'policy': policy_target,
                    'value': torch.tensor([[0.0]], dtype=torch.float32), # Placeholder
                    'rating': torch.tensor([[3.0]], dtype=torch.float32), # Placeholder
                    'agreement': 0.5 # Placeholder
                })
                
                # Allow time for opponent to move
                time.sleep(0.5)

            # Assign outcomes at end of game
            outcome = 1.0 # Let's pretend we won for the mock
            for exp in game_experiences:
                buffer.add(
                    exp['state'], exp['strategy'], exp['policy'],
                    exp['value'], exp['rating'], exp['agreement'], outcome
                )
                
            # Train model locally
            loss, metrics = trainer.train_step(batch_size=len(game_experiences))
            print(f"Game {game_idx} finished. Loss: {loss}")
            
            trainer.save_checkpoint()
            
        finally:
            # THIS ENSURES THE .WEBM IS ALWAYS SAVED PROPERLY
            adapter.close()

if __name__ == "__main__":
    run_online_loop()
