import os
from typing import Optional
import shutil

from utils import Logger
from config.getenv import GetEnv
from module.embed import EmbeddingPreprocessor

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

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
    ) -> QdrantVectorStore:
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
        




            