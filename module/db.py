import uuid
from typing import List, Dict
from config.getenv import GetEnv

# SQL Lite
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import sessionmaker, declarative_base

# Vector Store
from qdrant_client import QdrantClient, models

from module.LLMs import gpt

Base = declarative_base()
env = GetEnv()

class PlayBookMetadata(Base):
    __tablename__ = "playbook"
    entry_id = Column(String, primary_key=True, default=lambda : str(uuid.uuid4()))
    category = Column(String, nullable=False)
    content = Column(String, nullable=False)
    helpful_count = Column(Integer, default=0)
    harmful_count = Column(Integer, default=0)
    created_step = Column(Integer, default=0)
    last_used_step = Column(Integer, default=0)

sqlite_db_path = env.get_
engine = create_engine()
