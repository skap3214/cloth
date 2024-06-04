from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .extract import extract
from .config import Config
from .types import GraphChat, GraphAdd
import json

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
                'relations': [rel.model_dump() for rel in chunk['relations']]
            }
            yield json.dumps(chunk).encode()

    return StreamingResponse(stream())

@app.post("/chat")
def graph_chat(data: GraphChat):
    agent = Config.AGENT
    chat_stream = agent.chat(data.query, data.chat_type)

    def stream():
        for chunk in chat_stream:
            chunk = chunk.model_dump()
            yield json.dumps(chunk).encode()
    
    return StreamingResponse(stream())

