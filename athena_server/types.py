from pydantic import BaseModel
from typing import Optional, List, Literal
from cloth import Relation

class ChatResponse(BaseModel):
    sources: List[Relation]
    type: Literal['intermediate', 'output'] # To indicate what stage of the chain we are in
    delta: str

class GraphAdd(BaseModel):
    user_id: Optional[str] = None
    text: str
    init: bool = False

class GraphChat(BaseModel):
    query: str
    chat_type: Literal['node', 'edge', 'raw']