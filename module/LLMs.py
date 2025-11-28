from typing import Literal
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from config.getenv import GetEnv
from core.constant import SUPPORTED_PROVIDERS
from utils import Logger

env = GetEnv()
logger = Logger(__name__)

def normalize(value : str | None):
    if value is None:
        return None
    value = value.strip()
    return value if value else None

openai_api_key = normalize(env.get_openai_api_key)
claude_api_key = normalize(env.get_claude_api_key)
gemini_api_key = normalize(env.get_gemini_api_key)

openai_default_model = normalize(env.get_openai_model)
claude_default_model = normalize(env.get_claude_model)
gemini_default_model = normalize(env.get_gemini_model)

def validate_provider(provider: str):
    provider = provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Invalid provider '{provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return provider


def _create_openai_llm(model: str | None, temperature: float, **kwargs):
    if not openai_api_key:
        raise ValueError("OpenAI API key is missing.")

    target_model = normalize(model) or openai_default_model
    if not target_model:
        raise ValueError("OpenAI model name is missing.")

    return ChatOpenAI(
        api_key=openai_api_key,
        model=target_model,
        temperature=temperature,
        streaming=True,
        max_retries=10,
        **kwargs,
    )


def _create_anthropic_llm(model: str | None, temperature: float, **kwargs):
    if not claude_api_key:
        raise ValueError("Anthropic API key is missing.")

    target_model = normalize(model) or claude_default_model
    if not target_model:
        raise ValueError("Anthropic model name is missing.")

    return ChatAnthropic(
        api_key=claude_api_key,
        model=target_model,
        temperature=temperature,
        streaming=True,
        max_retries=10,
        **kwargs,
    )


def _create_google_llm(model: str | None, temperature: float, **kwargs):
    if not gemini_api_key:
        raise ValueError("Gemini API key is missing.")

    target_model = normalize(model) or gemini_default_model
    if not target_model:
        raise ValueError("Gemini model name is missing.")

    return ChatGoogleGenerativeAI(
        api_key=gemini_api_key,
        model=target_model,
        temperature=temperature,
        streaming=True,
        max_retries=10,
        **kwargs,
    )

@lru_cache(maxsize=20)
def get_llm(
    provider: Literal["openai", "anthropic", "google"] = "openai",
    model: str | None = None,
    temperature: float = 0.6,
    **kwargs,
):
    provider_key = validate_provider(provider)

    if provider_key == "openai":
        return _create_openai_llm(model, temperature, **kwargs)

    elif provider_key == "anthropic":
        return _create_anthropic_llm(model, temperature, **kwargs)

    elif provider_key == "google":
        return _create_google_llm(model, temperature, **kwargs)

    raise RuntimeError("Unexpected provider processing state.")