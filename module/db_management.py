import os
from typing import Any, Union, Optional

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
    def __init__(self, embedding_dir_or_repo_name : Optional[HuggingFaceEmbeddings] = None, **kwargs):
        self.embedding_model = self._init_embedding_model(embedding_dir_or_repo_name, **kwargs)
        self.vector_store_dir = env.get_vector_store_dir
    
    def _init_embedding_model(self, embedding_dir_or_repo_name: Optional[HuggingFaceEmbeddings], **kwargs):
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

    def to_disk(self,
                data : Document,
                db_name : Optional[str] = None,
                overwrite : bool = True,
                verbose : bool = True,
                ) -> QdrantVectorStore:
        if db_name is None:
            db_name = env.get_vector_store_name
        
        db_path = os.path.join(self.vector_store_dir, db_name)
        client = QdrantClient(path=db_path)

        if os.path.exists(db_path):
            if overwrite:
                if verbose:
                    logger.debug(f"Overwriting existing Qdrant collection : {db_name}")

                client.recreate_collection(
                    collection_name=db_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_model._client.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                vector_store = QdrantVectorStore.from_documents(
                    documents=data,
                    embedding=self.embedding_model,
                    path=db_path,
                    collection_name = db_name
                )
                if verbose:
                    logger.info(f"Collection {db_name} has been successfully owrtwritten.")
            
            else:
                if verbose:
                    logger.debug(f"Appeding documents to Qdrant collection : {db_name}")

                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=db_name,
                    embedding=self.embedding_model
                )
                vector_store.add_documents(data)

                if verbose:
                    logger.info(f"Collection {db_name} has been updated with new documents")
        
        else:
            if verbose:
                logger.debug(f"Creating new Qdrant collection : {db_name} at {db_path}")
            
            vector_store = QdrantVectorStore.from_documents(
                documents=data,
                embedding=self.embedding_model,
                path=db_path,
                collection_name=db_name
            )

            if verbose:
                logger.info(f"Collection {db_name} successfully created")
        
        return vector_store
    
    def from_disk(self,
                  db_name : Optional[str] = None
                  ) ->QdrantVectorStore:
        if db_name is None:
            db_name = env.get_vector_store_name
        
        db_path = os.path.join(env.get_vector_store_dir, db_name)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"The path {db_path} doen't exist")
        
        client = QdrantClient(path=db_path)

        collection = QdrantVectorStore(
            client=client,
            collection_name=db_name,
            embedding=self.embedding_model
        )

        return collection




            