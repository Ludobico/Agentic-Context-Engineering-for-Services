import uuid
from typing import List, Dict
from config.getenv import GetEnv

# SQL Lite
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import sessionmaker, declarative_base

# Vector Store
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant

from module.LLMs import gpt
from module.embed import EmbeddingPreprocessor

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

class VectorStoreManager:
    def __init__(self):
        self.embedding_model = EmbeddingPreprocessor.default_embedding_model()
        client = QdrantClient(path=env.get_vector_store_path)
        self.vector_store = Qdrant(
            client=client,
            collection_name=env.get_vector_store_dir,
            embeddings=self.embedding_model
        )

    async def upsert_entry(self, entry_id : str, content : str, payload : Dict):
        await self.vector_store.aadd_texts(
            texts=[content],
            metadatas = [payload],
            ids = [entry_id]
        )

    async def query(self, query_text : str, top_k : int) -> List[str]:
        docs = await self.vector_store.asimilarity_search(query=query_text, k=top_k)

        documents = [doc.metadata.get("entry_id") for doc in docs if 'entry_id' in doc.metadata]
        return documents
    
    async def delete_by_ids(self, ids : List[str]):
        if ids:
            await self.vector_store.adelete(ids=ids)

class DatabaseManager:
    def __init__(self):
        self.db = SessionLocal()
        self.vector_store = VectorStoreManager()
    
    def get_playbook_by_ids(self, ids : List[str]) -> List[Dict]:
        if not ids : return []

        entries = self.db.query(PlayBookMetadata).filter(PlayBookMetadata.entry_id.in_(ids)).all()
        playbook = [self._to_dict(entry) for entry in entries]
        return playbook
    
    async def retrieve_relevant_entries(self, query : str, top_k : int) -> List[Dict]:
        relevant_ids = await self.vector_store.query(query, top_k)
        return self.get_playbook_by_ids(relevant_ids)
    
    async def add_new_entry(self, entry_data : Dict):
        # sql에 메타데이터 저장
        new_entry = PlayBookMetadata(**entry_data)
        self.db.add(new_entry)
        self.db.commit()
        self.db.refresh(new_entry)

        # 벡터스토어에 벡터+페이로드 저장
        qdrant_payload = {"entry_id" : new_entry.entry_id}
        await self.vector_store.upsert_entry(new_entry.entry_id, new_entry.content, qdrant_payload)
        return self._to_dict(new_entry)
    
    def update_entry_stats(self, entry_id : str, tag : str, current_step : int):
        # curator를 통해 나온 entry의 id가 예전값과 같으면 업데이트
        entry = self.db.query(PlayBookMetadata).filter(PlayBookMetadata.entry_id == entry_id).first()
        
        if entry:
            if tag == 'helpful' : entry.helpful_count += 1
            elif tag == 'harmful' : entry.harmful_count += 1
            entry.last_used_step = current_step
            self.db.commit()
        
    def get_playbook_size(self) -> int:
        return self.db.query(PlayBookMetadata).count()
    
    def _to_dict(self, entry_obj) -> Dict:
        return {c.name : getattr(entry_obj, c.name) for c in entry_obj.__table__.columns}
    
    def close(self):
        self.db.close()
    


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