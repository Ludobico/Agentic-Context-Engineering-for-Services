from pydantic import BaseModel

class ChatRequest(BaseModel):
    query : str
    llm_provider : str
    llm_model : str