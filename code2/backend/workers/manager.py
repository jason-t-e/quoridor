import multiprocessing
import time
import logging

from online_game_loop import run_online_loop

logging.basicConfig(level=logging.INFO)

def start_worker(worker_id):
    logging.info(f"Worker {worker_id} started.")
    # Run the online game loop infinitely
    while True:
        try:
            # 1 game per loop iteration
            run_online_loop(num_games=1)
        except Exception as e:
            logging.error(f"Worker {worker_id} encountered an error: {e}")
            time.sleep(10) # Cool down before restarting

if __name__ == "__main__":
    NUM_WORKERS = 2
    
    processes = []
    
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(target=start_worker, args=(i,))
        p.start()
        processes.append(p)
        time.sleep(2) # Stagger start times
        
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logging.info("Shutting down workers...")
        for p in processes:
            p.terminate()
