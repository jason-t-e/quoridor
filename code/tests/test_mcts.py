from models.quoridor_net import QuoridorNet
from search.mcts import MCTS
from engine.game import QuoridorGame

def test_mcts_root_expansion():
    model = QuoridorNet()
    model.eval()
    mcts = MCTS(model, num_simulations=5)
    game = QuoridorGame()
    
    action_probs = mcts.search(game.board)
    assert len(action_probs) > 0
    assert abs(sum(action_probs.values()) - 1.0) < 1e-5
