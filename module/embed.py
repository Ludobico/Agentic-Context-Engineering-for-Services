import torch
import os
from utils import highlight_print
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from config.getenv import GetEnv

env = GetEnv()
access_key = env.get_huggingface_token

def get_torch_device(verbose : bool = False, **kwargs):
    """
    It will be useful for all functions that use the `pytorch` framework
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        highlight_print(f"device status : {device}", **kwargs)
    return device  

def get_cpu_device(verbose : bool = False, **kwargs):
    """
    Return the CPU device if CUDA is not used.
    """
    device = 'cpu'
    if verbose:
        highlight_print(f"device status : {device}", **kwargs)
    return device

class EmbeddingPreprocessor:   
    @staticmethod
    def default_embedding_model(download_path : Optional[os.PathLike] = None, use_gpu : bool = False ) -> HuggingFaceEmbeddings:
        """
        Embedding model to be used for the vector store.

        There are many embedding models available for free in Huggingface, and this function uses the following model.

        Model name : `google/embeddinggemma-300m`
        """
        if use_gpu:
            device = get_torch_device()
        else:
            device = get_cpu_device()

        model_kwargs = {
            "device" : device,
        }

        if access_key and access_key.strip():
            model_name = "google/embeddinggemma-300m"
            model_kwargs['token'] = access_key
        
        else:
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        embedding_model_name = HuggingFaceEmbeddings(model_name = model_name, cache_folder = download_path, encode_kwargs={"normalize_embeddings" : True}, model_kwargs=model_kwargs)
        return embedding_model_name
    
    @staticmethod
    def embedding_model(huggingface_model_name : str, download_path : Optional[os.PathLike] = None, use_gpu : bool = False) -> HuggingFaceEmbeddings:
        model_name = huggingface_model_name
        if use_gpu:
            device = get_torch_device()
        else:
            device = get_cpu_device()

        model_kwargs = {
            "device" : device,
            "token" : access_key
        }
        embedding_model_name = HuggingFaceEmbeddings(model_name = model_name, cache_folder = download_path, encode_kwargs={"normalize_embeddings" : True}, model_kwargs=model_kwargs)
        return embedding_model_name

if __name__ == "__main__":
    embedding = EmbeddingPreprocessor.default_embedding_model()