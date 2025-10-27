import uuid
from typing import List, Dict, Any
from config.getenv import GetEnv
from langchain_huggingface import HuggingFaceEmbeddings

from module.state import PlaybookEntry

# SQL Lite
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import sessionmaker, declarative_base

# Vector Store
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from utils import Logger

Base = declarative_base()
env = GetEnv()
logger = Logger(__name__)

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

class PlayBookVectorStore:
    def __init__(self):
        self.db_path = env.get_vector_store_path
        self.collection_name = env.get_vector_store_name

    def get_or_create_store(self, embedding_model : HuggingFaceEmbeddings) -> QdrantVectorStore:
        client = QdrantClient(path=self.db_path)

        # 컬렉션이 이미 존재하는지 확인
        try:
            client.get_collection(collection_name=self.collection_name)
        
        except Exception:
            # 컬렉션이 없으면 새로 생성
            embedding_size = embedding_model._client.get_sentence_embedding_dimension()
        
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=embedding_size,
                distance=models.Distance.COSINE
            )
        )

        vector_store = QdrantVectorStore(client = client, collection_name = self.collection_name, embeddings = embedding_model)
        return vector_store

    def add_playbook_entry(self, store : QdrantVectorStore, entry : PlaybookEntry):
        """
        새로운 playbook을 벡터스토어에 추가
        """
        
        page_cocntent = entry['content']

        metadata = {
            "entry_id" : entry['entry_id'],
            "category" : entry['category'],
            "helpful_count" : entry['helpful_count'],
            "harmful_count" : entry['harmful_count'],
            "created_step" : entry['created_step'],
            "last_used_step" : entry['last_used_step']
        }

        doc = Document(page_content=page_cocntent, metadata = metadata)

        store.add_documents([doc], ids={entry['entry_id']})
    
    def update_playbook_metadata(self, store : QdrantVectorStore, entry : PlaybookEntry):
        """
        기존의 메타데이터 항목 (helpful/harmful) 업데이트
        """
        
        entry_id = entry['entry_id']

        new_payload = {
            "entry_id": entry_id,
            "category": entry['category'],
            "helpful_count": entry['helpful_count'],
            "harmful_count": entry['harmful_count'],
            "created_step": entry['created_step'],
            "last_used_step": entry['last_used_step']
        }

        store.client.overwrite_payload(
            collection_name=self.collection_name,
            payload=new_payload,
            points=[entry_id]
        )

    def delete_playbook_entries(self, store : QdrantVectorStore, ids : List[str]):
        store.client.delete(collection_name=self.collection_name, points_selector=ids)
