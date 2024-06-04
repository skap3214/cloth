from cloth.neo4j_graph import Neo4jGraphstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .chat import Agent
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings


@dataclass
class Config:
    GRAPH = Neo4jGraphstore(
        collection_name='athena',
        embeddings_model=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    AGENT = Agent(
        graph=GRAPH
    )

    SPLITTER = RecursiveCharacterTextSplitter(
        keep_separator=False,
        chunk_size=500,
        chunk_overlap=30
    )
