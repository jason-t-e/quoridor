import argparse
import yaml
import time
from utils.model_manager import ModelManager
from adapters.example_site_adapter import ExampleSiteAdapter

def main():
    parser = argparse.ArgumentParser(description="Play Quoridor Online")
    parser.add_argument("--config", type=str, default="configs/settings.yaml", help="Path to settings file")
    args = parser.parse_args()

    # Load Settings
    with open(args.config, 'r') as f:
        settings = yaml.safe_load(f)

    print(f"Loaded settings from {args.config}")

    # Manage Models
    model_manager = ModelManager(args.config)
    model_manager.cleanup_old_models()
    
    # Normally we would load the model here:
    # model = model_manager.load_active_model(QuoridorNet)
    print(f"Active model: {model_manager.get_active_model_path()}")

    # Initialize Adapter
    adapter = ExampleSiteAdapter(settings)
    
    try:
        adapter.connect()
        
        games_to_play = settings['online_play'].get('games_to_play', 1)
        save_interval = settings.get('models', {}).get('save_interval', 100)
        super_save_interval = settings.get('models', {}).get('super_save_interval', 1000)
        
        # We assume model_manager loaded the total_games played from the latest checkpoint
        total_games_played = getattr(model_manager, 'total_games_played', 0)
        
        for game_num in range(1, games_to_play + 1):
            total_games_played += 1
            print(f"--- Starting Game {game_num} (Total lifetime games: {total_games_played}) ---")
            
            game_experiences = [] # Store (state, action, reward)
            
            while not adapter.is_game_over():
                if adapter.is_my_turn():
                    state = adapter.get_board_state()
                    # action = model.predict(state) # Use the loaded model here
                    action = "dummy_action" # Placeholder
                    
                    adapter.make_move(action)
                    
                    # Store experience
                    # game_experiences.append((state, action))
                    
                time.sleep(1) # Polling delay
                
            # Game finished, calculate reward and train incrementally
            # final_reward = adapter.get_game_result()
            # model.train(game_experiences, final_reward)
            print(f"--- Finished Game {game_num} ---")
            
            # Checkpoint Logic
            if total_games_played % save_interval == 0:
                print(f"Reached {save_interval} games save point! Saving checkpoint...")
                # model.save(f"quoridor_net_ckpt_{total_games_played}.pt")
                # Update settings or model manager so next session resumes from here
            
            if total_games_played % super_save_interval == 0:
                print(f"Reached {super_save_interval} games SUPER save point! Archiving checkpoint...")
                # model.save(f"quoridor_net_super_ckpt_{total_games_played}.pt")
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        adapter.close()
        print("Adapter closed. Goodbye.")

if __name__ == "__main__":
    main()
