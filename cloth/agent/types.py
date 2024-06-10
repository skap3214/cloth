from typing import List
from ..graph.types import Relation
from pydantic import BaseModel

class ChatResponse(BaseModel):
    sources: List[Relation]
    type: str
    delta: str