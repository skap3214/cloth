from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .extract import extract
from .config import Config
from .types import GraphChat, GraphAdd
from cloth.utils.logger import get_logger
import json

logger = get_logger("athena")

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
    if not data.user_id:
        ValueError("UserId not found, please include a valid userId")

    metadata = {
        'user_id': data.user_id,
        'graph_id': data.graph_id
    }
    if data.init:
        Config.GRAPH.reset(metadata=metadata)
    
    documents = extract(data.text)
    add_stream = Config.GRAPH.add(documents, stream=data.stream, metadata=metadata)

    if data.stream:
        def stream():
            for chunk in add_stream:
                chunk = {
                    'document': {'page_content': chunk['document'].page_content, 'metadata': chunk['document'].metadata},
                    'relations': [rel.model_dump() for rel in chunk['relations']]
                }
                yield json.dumps(chunk).encode()

        return StreamingResponse(stream())
    else:
        def stream():
            for document, rel_list in zip(add_stream['documents'], add_stream['relations']):
                chunk = {
                    'document': {'page_content': document.page_content, 'metadata': document.metadata},
                    'relations': [rel.model_dump() for rel in rel_list]
                }
                yield json.dumps(chunk).encode()

        return StreamingResponse(stream())

@app.post("/chat")
def graph_chat(data: GraphChat):
    agent = Config.AGENT
    metadata = {
        'user_id': data.user_id,
        'graph_id': data.graph_id
    }
    chat_stream = agent.chat(data.query, data.chat_type, metadata=metadata)

    def stream():
        for chunk in chat_stream:
            chunk = chunk.model_dump()
            print(chunk['delta'])
            yield json.dumps(chunk).encode()
    
    return StreamingResponse(stream(), media_type="application/json")

