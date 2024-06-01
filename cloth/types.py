from pydantic import BaseModel, model_validator
from cloth.utils.id import generate_id
from typing import Optional

class Node(BaseModel):
    id: Optional[str] = None
    name: str
    type: str = "Node"

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

    @model_validator(mode='before')
    def set_id(cls, values):
        print(values)
        if not values.get("edge").get("id"):
            node_1 = values['node_1']['name']
            node_2 = values['node_2']['name']
            values['edge']['id'] = generate_id(f"{node_1}_{values['edge']['name']}_{node_2}")
        return values