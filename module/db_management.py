import os
from typing import Optional, Literal
import shutil
import gc

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

def ensure_datetime(v):
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        return datetime.fromisoformat(v)

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
        if embedding_dir_or_repo_name is None:
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
            self.client.get_collection(self.db_name)
            if verbose:
                logger.debug(f"Appending documents to existing Qdrant collection: {self.db_name}")

        except Exception:
            if verbose:
                logger.debug(f"Creating new Qdrant collection: {self.db_name} at {self.db_path}")
            
            try:
                embedding_size = self.embedding_model._client.get_sentence_embedding_dimension()
            except AttributeError:
                try:
                    embedding_size = self.embedding_model._client.get_sentence_embedding_dimension()
                except AttributeError:
                    logger.warning("Could not find get_sentence_embedding_dimension(). Inferring size from dummy text.")
                    embedding_size = len(self.embedding_model.embed_query("test"))

            self.client.create_collection(
                collection_name=self.db_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                )
            )
            if verbose:
                 logger.info(f"Collection '{self.db_name}' created at {self.db_path}")

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.db_name,
            embedding=self.embedding_model,
        )

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
                        key='metadata.entry_id',
                        match=models.MatchValue(value=entry_id)
                    ) for entry_id in entry_ids
                ]
            ),
            wait=True
        )
        logger.info(f"Delete {len(entry_ids)} entries from vector store")

    def get_entry_by_id(self, entry_id : str) -> dict | None:
        records, _ = self.client.scroll(
            collection_name=self.db_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.entry_id",
                        match=models.MatchValue(value=entry_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if not records:
            return None
        
        point = records[0]
        return point.payload
    
    def get_all_entries(self) -> list[dict]:
        all_records = []
        next_page_offset = None

        while True:
            records, next_page_offset = self.client.scroll(
                collection_name=self.db_name,
                scroll_filter=None,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=next_page_offset
            )
        
            if not records:
                break

            all_records.extend([r.payload for r in records])

            if next_page_offset is None:
                break
                
        return all_records


        
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
            created_at = ensure_datetime(entry['created_at']),
            updated_at = ensure_datetime(entry['updated_at']),
            last_used_at = ensure_datetime(entry.get('last_used_at'))
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
        _vector_store_instance = VectorStore(use_gpu=env.get_embedding_gpu)
    return _vector_store_instance

def verify_vectorstore_db_sync(verbose: bool = True) -> bool:
    db = get_db_instance()
    vector_store = get_vector_store_instance()

    db_entries = db.get_all_entries()
    vs_entries = vector_store.get_all_entries()

    vs_dict = {e['metadata']['entry_id']: e for e in vs_entries if 'metadata' in e}

    all_match = True

    for entry in db_entries:
        entry_id = entry['entry_id']
        if entry_id not in vs_dict:
            all_match = False
            if verbose:
                print(f"[MISSING IN VECTORSTORE] entry_id: {entry_id}")
            continue

        vs_entry = vs_dict[entry_id]['metadata']
        fields_to_check = ['category', 'helpful_count', 'harmful_count']

        for field in fields_to_check:
            db_value = entry.get(field)
            vs_value = vs_entry.get(field)

            if db_value != vs_value:
                all_match = False
                if verbose:
                    print(f"[MISMATCH] entry_id: {entry_id}, field: {field}, DB: {db_value}, VS: {vs_value}")

    if all_match and verbose:
        print("All entries match between DB and VectorStore.")
    elif verbose:
        print("Some entries do not match.")

    return all_match

def close_db():
    global _db_instance
    if _db_instance is not None:
        _db_instance.engine.dispose()
        _db_instance = None

def close_vector_store():
    global _vector_store_instance
    if _vector_store_instance is not None:
        _vector_store_instance.client.close()
    _vector_store_instance = None

def reset_all_stores(target : Literal['db', 'vs', 'both'] = 'both'):
    if target in ("db", "both"):
        close_db()
    if target in ("vs", "both"):
        close_vector_store()
    
    gc.collect()

    if target in ("db", "both"):
        db_path = env.get_db_path
        if os.path.exists(db_path):
            os.remove(db_path)
    
    if target in ("vs", "both"):
        qdrant_path = os.path.join(env.get_vector_store_dir, env.get_vector_store_name)
        if os.path.exists(qdrant_path):
            shutil.rmtree(qdrant_path)
    

if __name__ == "__main__":
    reset_all_stores()
