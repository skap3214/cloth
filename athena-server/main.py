from typing import Optional, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from .extract import extract
from .config import Config
import json

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

    def stream():
        for chunk in add_stream:
            chunk = {
                'document': {'page_content': chunk['document'].page_content, 'metadata': chunk['document'].metadata},
                'relatons': [rel.model_dump() for rel in chunk['relations']]
            }
            yield json.dumps(chunk).encode()

    return StreamingResponse(stream())

@app.post("/chat")
def graph_chat(data: GraphChat):
    pass