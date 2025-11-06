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
from datetime import datetime

env = GetEnv()
logger = Logger(__name__)

_db_instance = None
_vector_store_instance = None

class VectorStore:
    """
    Manages a Qdrant vector store for document embeddings.

    This class handles the initialization of the embedding model and the Qdrant client,
    providing methods to add documents to the store and retrieve the store instance.

    Args:
        embedding_dir_or_repo_name (Optional[str]): 
            The local directory path or Hugging Face repository name for the embedding model.
            If None, it attempts to use a default model specified in `EmbeddingPreprocessor`,
            which typically requires a Hugging Face token to be available in the environment.
            Defaults to None.
        db_name (Optional[str]): 
            The name of the Qdrant collection (database).
            If None, it falls back to the name specified by the `get_vector_store_name`
            environment variable. Defaults to None.
        **kwargs: 
            Additional keyword arguments that are passed directly to the HuggingFaceEmbeddings
            model constructor, allowing for custom model configuration.
    """
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
    @property
    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        return self.embedding_model
    
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
            # 1. 컬렉션이 존재하는지 확인
            self.client.get_collection(self.db_name)
            if verbose:
                logger.debug(f"Appending documents to existing Qdrant collection: {self.db_name}")

        except Exception: # ValueError 또는 qdrant_client.http.exceptions.UnexpectedResponse 등
            # 2. 컬렉션이 없으면 새로 생성
            if verbose:
                logger.debug(f"Creating new Qdrant collection: {self.db_name} at {self.db_path}")
            
            # 임베딩 모델의 차원(dimension)을 확인합니다.
            try:
                # HuggingFaceEmbeddings의 경우 .client에 실제 모델이 있을 수 있음
                embedding_size = self.embedding_model._client.get_sentence_embedding_dimension()
            except AttributeError:
                # 또는 _client (기존 코드 존중)
                try:
                    embedding_size = self.embedding_model._client.get_sentence_embedding_dimension()
                except AttributeError:
                    # 그래도 실패하면, "test" 문자열을 임베딩하여 차원 추론
                    logger.warning("Could not find get_sentence_embedding_dimension(). Inferring size from dummy text.")
                    embedding_size = len(self.embedding_model.embed_query("test"))

            # self.client (단일 인스턴스)를 사용하여 컬렉션 생성
            self.client.create_collection(
                collection_name=self.db_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE, # 또는 models.Distance.DOT 등 필요에 따라
                )
            )
            if verbose:
                 logger.info(f"Collection '{self.db_name}' created at {self.db_path}")

        # 3. LangChain 래퍼(wrapper) 생성
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.db_name,
            embedding=self.embedding_model,
        )

        # 4. [중요] from_documents(...) 대신 add_documents(...) 사용
        # path= 인자를 넘기지 않으므로 새 클라이언트를 생성하려 시도하지 않아 런타임 오류가 발생하지 않습니다.
        # 컬렉션이 있든 없든(방금 만들었든) 문서를 추가합니다.
        vector_store.add_documents(data)

        if verbose:
            logger.info(f"Collection '{self.db_name}' successfully updated with {len(data)} new documents.")
    
    def from_disk(self) -> QdrantVectorStore:

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
        
    def delete_by_entry_ids(self, entry_ids : list[str]):
        if not entry_ids:
            return
        
        self.client.delete(
            collection_name=self.db_name,
            points_selector=models.Filter(
                should=[
                    models.FieldCondition(
                        key='entry_id',
                        match=models.MatchValue(value=entry_id)
                    ) for entry_id in entry_ids
                ]
            ),
            wait=True
        )
        logger.info(f"Delete {len(entry_ids)} entries from vector store")
        
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

def get_db_instance() -> PlayBookDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = PlayBookDB()
    return _db_instance

def get_vector_store_instance() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(use_gpu=True)
    return _vector_store_instance

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
