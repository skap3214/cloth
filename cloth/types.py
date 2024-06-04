from pydantic import BaseModel, model_validator
from cloth.utils.id import generate_id
from typing import Optional, Dict
from typing_extensions import Self

class Node(BaseModel):
    id: Optional[str] = None
    name: str
    type: str = "Node"
    metadata: Optional[Dict] = {}

    @model_validator(mode='before')
    def set_id(cls, values):
        if 'id' not in values:
            values['id'] = generate_id(values.get('name', ''))
        return values
    
    @model_validator(mode='before')
    def normalize_type(cls, values):
        if 'type' not in values:
            return values
        else:
            values['type'] = values['type'].title().replace(" ", "_")
            return values

class Edge(BaseModel):
    id: Optional[str] = None
    name: str
    type: str = "Edge"
    metadata: Optional[Dict] = {}

    @model_validator(mode='before')
    def normalize_type(cls, values):
        if 'type' not in values:
            return values
        else:
            values['type'] = values['type'].title().replace(" ", "_")
            return values
    
class Relation(BaseModel):
    node_1: Node
    edge: Edge
    node_2: Node

    @model_validator(mode='after')
    def check_id(self) -> Self:
        if not self.edge.id:
            self.edge.id = generate_id(f"{self.node_1.name}_{self.edge.name}_{self.node_2.name}")
        return self