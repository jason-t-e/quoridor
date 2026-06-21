import uuid
from typing import List
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from .database import SessionLocal, Game, MoveRecord, GameMode

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.get("/api/games/active")
def get_active_games(db: Session = Depends(get_db)):
    games = db.query(Game).filter(Game.status == "ACTIVE").all()
    return games

@router.get("/api/games/{game_id}/replay")
def get_game_replay(game_id: str, db: Session = Depends(get_db)):
    moves = db.query(MoveRecord).filter(MoveRecord.game_id == game_id).order_by(MoveRecord.turn_number).all()
    return moves

@router.post("/api/games")
def create_game(mode: str, player_1: str, player_2: str, db: Session = Depends(get_db)):
    game = Game(
        id=str(uuid.uuid4()),
        mode=GameMode[mode],
        player_1=player_1,
        player_2=player_2
    )
    db.add(game)
    db.commit()
    db.refresh(game)
    return game

@router.websocket("/ws/games/active")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open, in reality we broadcast updates from workers to this WS
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
