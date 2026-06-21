import pytest
import os
from adapters.playwright_adapter import PlaywrightAdapter
from game_engine.moves import PawnMove, WallMove

# We test that the adapter can instantiate and close without throwing errors.
# We skip the live playwright `start` check to avoid needing an actual browser in CI.

def test_playwright_adapter_init():
    adapter = PlaywrightAdapter(record_dir="data/test_recordings")
    assert adapter.headless == True
    assert adapter.record_dir == "data/test_recordings"
    assert os.path.exists("data/test_recordings")
    adapter.close()

def test_playwright_execute_moves():
    adapter = PlaywrightAdapter()
    
    pawn_move = PawnMove(player=0, to_row=1, to_col=4)
    wall_move = WallMove(player=0, row=1, col=2, orientation='h')
    
    # Should not throw errors even if page is None for dummy mode
    try:
        adapter.execute_move(pawn_move)
        adapter.execute_move(wall_move)
    except Exception as e:
        pytest.fail(f"execute_move raised an exception: {e}")
        
    adapter.close()
