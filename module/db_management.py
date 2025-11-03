from typing import Any, Union, Optional

from utils import Logger
from config.getenv import GetEnv
from module.embed import EmbeddingPreprocessor

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

env = GetEnv()

class VectorStore:
    huggingface_token = env.get_huggingface_token
    def __init__(self, embedding_dir_or_repo_name : Optional[HuggingFaceEmbeddings] = None, **kwargs):
        if embedding_dir_or_repo_name is None and self.huggingface_token:
            self.embedding_dir = EmbeddingPreprocessor.default_embedding_model(**kwargs)
        elif embedding_dir_or_repo_name is not None:
            self.embedding_dir = EmbeddingPreprocessor.embedding_model(embedding_dir_or_repo_name, **kwargs)
        elif embedding_dir_or_repo_name is None and self.huggingface_token is None:
            raise ValueError("Either a Huggingface token or an embedding directory or repo name must be provided.")
        else:
            raise ValueError("Invalid initialization paramters for VectorStore")
            