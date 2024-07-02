from pydantic import BaseModel, model_validator
from cloth.utils.id import generate_id
from typing import Optional, Dict, List
from typing_extensions import Self

class Node(BaseModel):
    id: Optional[str] = None
    name: str
    type: str = "Node"
    metadata: Optional[Dict] = {}

    @model_validator(mode='after')
    def set_id(self) -> Self:
        if not self.id:
            self.id = generate_id(**{**self.metadata, 'name': self.name})
        return self
    
    @model_validator(mode='after')
    def normalize_type(self) -> Self:
        if self.type:
            return self
        else:
            self.type = self.type.title().replace(" ", "_")
            return self

class Edge(BaseModel):
    id: Optional[str] = None
    name: str
    type: str = "Edge"
    metadata: Optional[Dict] = {}

    @model_validator(mode='after')
    def normalize_type(self) -> Self:
        if not self.type:
            return self
        else:
            self.type = self.type.title().replace(" ", "_")
            return self
    
class Relation(BaseModel):
    node_1: Node
    edge: Edge
    node_2: Node

    @model_validator(mode='after')
    def check_id(self) -> Self:
        if not self.edge.id:
            self.edge.id = generate_id(node_1=self.node_1.id, edge=self.edge.name, node_2=self.node_2.id, **self.edge.metadata)
        return self
