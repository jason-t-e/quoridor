import time
from engine.game import QuoridorGame

def test_1000_random_games():
    start_time = time.time()
    
    games_to_play = 1000
    total_moves = 0
    
    for i in range(games_to_play):
        game = QuoridorGame()
        game.play_random_game()
        total_moves += game.board.turn

    duration = time.time() - start_time
    games_per_sec = games_to_play / duration
    moves_per_sec = total_moves / duration
    
    print(f"\nPlayed {games_to_play} random games in {duration:.2f} seconds.")
    print(f"Games/sec: {games_per_sec:.2f}")
    print(f"Moves/sec: {moves_per_sec:.2f}")
    
    # If it completes without raising exceptions (RuntimeError for zero moves), it passed.
    assert games_to_play > 0

if __name__ == "__main__":
    test_1000_random_games()
