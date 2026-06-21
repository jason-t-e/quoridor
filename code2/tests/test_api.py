import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.api.database import Base, engine, SessionLocal

# Recreate the database for tests
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_create_and_get_active_games():
    # Create game
    response = client.post("/api/games?mode=ONLINE&player_1=Bot_v1&player_2=Human")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "ONLINE"
    assert "id" in data
    
    # Get active games
    response = client.get("/api/games/active")
    assert response.status_code == 200
    games = response.json()
    assert len(games) >= 1
    
def test_websocket_connection():
    with client.websocket_connect("/ws/games/active") as websocket:
        # Just testing connection works
        assert websocket is not None
