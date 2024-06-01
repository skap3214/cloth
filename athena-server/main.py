from typing import Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from .extract import extract
from .config import Config

# Types

class GraphAdd(BaseModel):
    user_id: Optional[str] = None
    text: str
    init: bool = False

class GraphChat(BaseModel):
    query: str
    chat_type: Literal['node', 'edge', 'raw']

#######

app = FastAPI()

@app.get("/")
def root():
    return "Athena API"

@app.post("/add")
def graph_add(data: GraphAdd):
    if data.init:
        Config.GRAPH.reset()
    
    # Not checking for user_id yet
    documents = extract(data.text)
    add_stream = Config.GRAPH.add(documents, stream=True)

    return StreamingResponse(add_stream)

@app.post("/chat")
def graph_chat(data: GraphChat):
    pass