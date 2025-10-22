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

db_path = f"sqlite:///{env.get_db_path}"
engine = create_engine(db_path)
SessionLocal = sessionmaker(autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)



def test_sql():
    session = SessionLocal()
    new_item = PlayBookMetadata(
        category="테스트 카테고리",
        content="이건 테스트 콘텐츠입니다.",
        helpful_count=1,
        harmful_count=0
    )

    session.add(new_item)
    session.commit()

    print("데이터 추가 완료")
    result = session.query(PlayBookMetadata).filter_by(category="테스트 카테고리").first()

    print(result.content)

if __name__ == "__main__":
    test_sql()