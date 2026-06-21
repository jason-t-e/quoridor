from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum
import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./quoridor_data.db"
# If we were using Postgres in docker:
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@db/quoridor"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class GameMode(enum.Enum):
    ONLINE = "ONLINE"
    SELF_PLAY = "SELF_PLAY"
    HUMAN = "HUMAN"

class GameStatus(enum.Enum):
    ACTIVE = "ACTIVE"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

class Game(Base):
    __tablename__ = "games"

    id = Column(String, primary_key=True, index=True)
    mode = Column(Enum(GameMode))
    player_1 = Column(String)
    player_2 = Column(String)
    winner = Column(String, nullable=True)
    status = Column(Enum(GameStatus), default=GameStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class MoveRecord(Base):
    __tablename__ = "moves"

    id = Column(String, primary_key=True, index=True)
    game_id = Column(String, index=True)
    turn_number = Column(Integer)
    player = Column(String)
    move_notation = Column(String)
    board_state_after = Column(JSON)
    thinking_time_ms = Column(Integer)
    evaluation = Column(JSON, nullable=True)

class ModelVersion(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    metrics = Column(JSON)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)
