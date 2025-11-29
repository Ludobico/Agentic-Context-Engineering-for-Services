from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    query : str
    llm_provider : str
    llm_model : str
    session_id : str
    execution_mode : Optional[str] = 'standard'