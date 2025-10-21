from config.getenv import GetEnv
from langchain_openai import ChatOpenAI

env = GetEnv()
openai_api_key = env.get_openai_api_key

def gpt(temperature : float = 0.1, model : str = 'gpt-4o-mini', streaming : bool = True):
    """
    Args:
    temperature (float, optional): The temperature parameter for the chat model. Defaults to 0.1.
    model (str, optional): The name of the model.
    streaming (bool, optional): Whether to use streaming mode for the chat model. Defaults to True.

    Returns:
        ChatOpenAI: The fine-tuned chat model.
    """
    llm = ChatOpenAI(temperature=temperature, api_key=openai_api_key, model=model, streaming=streaming)
    return llm