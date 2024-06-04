from cloth.neo4j_graph import Neo4jGraphstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .chat import Agent

class Config:
    AGENT = Agent(
    )

    GRAPH = AGENT.graph

    SPLITTER = RecursiveCharacterTextSplitter(
        keep_separator=False,
        chunk_size=500,
        chunk_overlap=30
    )
