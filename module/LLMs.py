from typing import Literal
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from config.getenv import GetEnv
from utils import Logger

env = GetEnv()
logger = Logger(__name__)
openai_api_key = env.get_openai_api_key
claude_api_key = env.get_claude_api_key
gemini_api_key = env.get_gemini_api_key

def _create_openai_llm(
        model : str,
        temperature : float = 0.6,
        **kwargs
):
    target_model = model if model else env.get_openai_model
    return ChatOpenAI(
        api_key=openai_api_key,
        model=target_model,
        temperature=temperature,
        streaming=True,
        max_retries=10,
        **kwargs
    )

def _create_anthropic_llm(
        model : str,
        temperature : float = 0.6,
        **kwargs
):
    target_model = model if model else env.get_claude_model
    return ChatAnthropic(
        api_key=claude_api_key,
        model = target_model,
        temperature=temperature,
        streaming=True,
        max_retries=10,
        **kwargs
    )

def _create_google_llm(
        model : str,
        temperature : float = 0.6,
        **kwargs
):
    target_model = model if model else env.get_gemini_model
    return ChatGoogleGenerativeAI(
        google_api_key = gemini_api_key,
        model=target_model,
        temperature = temperature,
        streaming = True,
        max_retries=10,
        **kwargs
    )

@lru_cache(maxsize=20)
def get_llm(
    provider : Literal["openai", "anthropic", "google"] = 'openai',
    model : str = None,
    temperature : float = 0.6,
    **kwargs
):
    provider_key = provider.lower().strip() if provider else "openai"
    if provider_key == "anthropic":
        return _create_anthropic_llm(model, temperature, **kwargs)
    
    elif provider_key == "google":
        return _create_google_llm(model, temperature, **kwargs)
    # Default & OpenAI Case
    else:
        if provider_key != "openai":
            logger.warning(f"Warning: Unknown provider '{provider}'. Falling back to OpenAI.")
            
        return _create_openai_llm(model, temperature, **kwargs)