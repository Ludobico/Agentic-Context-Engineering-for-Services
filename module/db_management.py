import os
from typing import Optional
import shutil

from utils import Logger
from config.getenv import GetEnv
from module.embed import EmbeddingPreprocessor
from core.state import PlaybookEntry

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, DateTime, insert, select, update, delete
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from datetime import datetime

env = GetEnv()
logger = Logger(__name__)


class VectorStore:
    huggingface_token = env.get_huggingface_token
    def __init__(self,
                 embedding_dir_or_repo_name : Optional[str] = None,
                 db_name : Optional[str] = None,
                 **kwargs
                 ):
        self.embedding_model = self._init_embedding_model(embedding_dir_or_repo_name, **kwargs)
        self.vector_store_dir = env.get_vector_store_dir
        self.db_path, self.db_name = self._get_db_info(db_name)
        self.client = QdrantClient(path=self.db_path)
    
    def _init_embedding_model(self, embedding_dir_or_repo_name: Optional[str], **kwargs) -> HuggingFaceEmbeddings:
        if embedding_dir_or_repo_name is None and self.huggingface_token:
            return EmbeddingPreprocessor.default_embedding_model(**kwargs)
        elif embedding_dir_or_repo_name is not None:
            return EmbeddingPreprocessor.embedding_model(embedding_dir_or_repo_name, **kwargs)
        elif embedding_dir_or_repo_name is None and self.huggingface_token is None:
            raise ValueError(
                "Either a Hugging Face token or an embedding directory or repo name must be provided."
            )
        else:
            raise ValueError("Invalid initialization parameters for VectorStore.")
    
    def _get_db_info(self, db_name : Optional[str] = None) -> str:
        if db_name is None:
            db_name = env.get_vector_store_name
        
        db_path = os.path.join(self.vector_store_dir, db_name)
        return db_path, db_name

    def to_disk(
        self,
        data: list[Document],
        verbose: bool = True,
    ):
        """
        Append documents to an existing local Qdrant vector store.
        If the store does not exist, create a new one.
        """
        try:
            self.client.get_collection(self.db_name)
            collection_exists = True
        except ValueError:
            collection_exists = False

        if collection_exists:
            if verbose:
                logger.debug(f"Appending documents to existing Qdrant collection: {self.db_name}")

            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.db_name,
                embedding=self.embedding_model,
            )

            vector_store.add_documents(data)

            if verbose:
                logger.info(f"Collection '{self.db_name}' updated with {len(data)} new documents.")

        else:
            if verbose:
                logger.debug(f"Creating new Qdrant collection: {self.db_name} at {self.db_path}")

            self.client.create_collection(
                collection_name=self.db_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_model._client.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                )
            )

            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.db_name,
                embedding=self.embedding_model,
            )
            vector_store.from_documents(
                documents=data,
                embedding=self.embedding_model,
                path=self.db_path,
                collection_name=self.db_name, 
            )

            if verbose:
                logger.info(f"Collection '{self.db_name}' successfully created with {len(data)} documents.")
    
    def from_disk(self) ->QdrantVectorStore:

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"The path {self.db_path} doen't exist")

        collection = QdrantVectorStore(
            client=self.client,
            collection_name=self.db_name,
            embedding=self.embedding_model
        )

        return collection
    
    def get_doc_count(self) -> int:
        try:
            count_result = self.client.count(collection_name=self.db_name, exact=True)
            return count_result.count
        
        except (ValueError) as e:
            logger.warning(f"Could not count docs in {self.db_name} (collection may not exist) : {e}")
            return 0
        
class PlayBookDB:
    def __init__(self):
        self.db_path = env.get_db_path
        self.engine : Engine = create_engine(f"sqlite:///{self.db_path}", echo=False, future=True)
        self.metadata = MetaData()

        self.playbook = Table(
            "playbook",
            self.metadata,
            Column("entry_id", String, primary_key=True),
            Column("category", String, nullable=False),
            Column("content", String, nullable=False),
            Column("helpful_count", Integer, default=0),
            Column("harmful_count", Integer, default=0),
            Column("created_at", DateTime, nullable=False),
            Column("updated_at", DateTime, nullable=False),
            Column("last_used_at", DateTime, nullable=True),
        )

        self.metadata.create_all(self.engine)
    
    def add_entry(self, entry: PlaybookEntry):
        stmt = insert(self.playbook).values(
            entry_id = entry['entry_id'],
            category = entry['category'],
            content = entry['content'],
            helpful_count = entry['helpful_count'],
            harmful_count = entry['harmful_count'],
            created_at = entry['created_at'],
            updated_at = entry['updated_at'],
            last_used_at = entry.get('last_used_at')
        ).prefix_with("OR REPLACE")

        with self.engine.begin() as conn:
            conn.execute(stmt)
    

    def get_entry(self, entry_id : str) -> PlaybookEntry | None:
        stmt = select(self.playbook).where(self.playbook.c.entry_id == entry_id)
        with self.engine.connect() as conn:
            result = conn.execute(stmt).mappings().first()
            return dict(result) if result else None
        
    def get_all_entries(self) -> list[PlaybookEntry]:
        stmt = select(self.playbook)
        with self.engine.connect() as conn:
            results = conn.execute(stmt).mappings().all()
            return [dict(r) for r in results]
    
    def delete_entry(self, entry_id : str):
        stmt = delete(self.playbook).where(self.playbook.c.entry_id == entry_id)
        with self.engine.begin() as conn:
            conn.execute(stmt)
    
    def update_last_used(self, entry_id : str, timestamp : datetime):
        stmt = (
            update(self.playbook)
            .where(self.playbook.c.entry_id == entry_id)
            .values(last_used_at = timestamp, updated_at = datetime.now())
        )
        with self.engine.begin() as conn:
            conn.exec_driver_sql(stmt)

if __name__ == "__main__":
    db = PlayBookDB()

    entry = {
        "entry_id": "pbk_001",
        "category": "motivation",
        "content": "Focus and be consistent.",
        "helpful_count": 5,
        "harmful_count": 0,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }

    db.add_entry(entry)
