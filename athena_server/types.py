from pydantic import BaseModel
from typing import Optional, List, Literal
from cloth import Relation

class ChatResponse(BaseModel):
    sources: List[Relation]
    type: Literal['intermediate', 'output'] # To indicate what stage of the chain we are in
    delta: str

class GraphAdd(BaseModel):
    user_id: str
    graph_id: Optional[str] = "1" # Make this None once we add multiple graphs per user
    text: str
    init: bool = False
    stream: bool = True

class GraphChat(BaseModel):
    query: str
    chat_type: Literal['node', 'edge', 'raw']
    user_id: str
    graph_id: Optional[str] = "1"