import sys
sys.path.append(".")

from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from cloth import Neo4jVectorGraphstore, Relation

collection_name = "test"
vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory="./local",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)
node_type = [
    "object", "entity", "location", "organization", 
    "person", "condition", "acronym", "documents", 
    "service", "concept", "emotion", "etc..."
]

uri = "bolt://localhost:7687"
user = "neo4j"
cloth = Neo4jVectorGraphstore(
    collection_name=collection_name,
    node_type=node_type,
    vectorstore=vectorstore,
    # neo4j_password=password,
    neo4j_uri=uri,
    neo4j_user=user,
    # llm=ChatOllama(model='llava-llama3:latest', temperature=0.1),
    llm=ChatGroq(model='llama3-70b-8192', temperature=0.0),
    # llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0.1)
)



# Prepare
from textwrap import dedent
# Article about AI

# filename = "examples/data/ai.txt"
# sep = ">>"

# Manager Reviews
filename = "examples/data/reviews.txt"
sep = "\n"

with open(filename, "r") as f:
    text = f.read()

documents = [Document(page_content=ex.strip()) for ex in text.split(sep)]


# Remove/Reset everything
graph_metadata = None
# output = cloth.reset({})

# Add documents
# output = cloth.add(documents)


# Searches
def sim_search(query):
    docs = cloth.similarity_search(
        query,
        filter=graph_metadata
    )
    print(f"Sim Search: {docs}")

def edge_search(query):
    doc_type = "edge"
    nodes = cloth.similarity_search(
        query, 
        doc_type=doc_type, 
        k=1,
        filter=graph_metadata
    )[0]
    output = cloth.get_adjacent_edges(node_name=nodes)
    return output

def node_search(query):
    doc_type = "node"
    nodes = cloth.similarity_search(
        query, 
        doc_type=doc_type, 
        k=1,
        filter=graph_metadata
    )[0]
    output = cloth.get_adjacent_edges(node_name=nodes)
    return output

def path_search(query):
    # node_1, node_2 = cloth.similarity_search(query, doc_type='node', k=2)[:2]
    node_1, node_2 = "Bitter Lesson", "Search"
    output = cloth.find_paths(node_1, node_2)
    return output

# Example usage
# output = sim_search("What is AI?")
# output = edge_search("What is AI?")
output = node_search("Who is Bob?")
# output = path_search("AI and Chess")

incoming = output['incoming']
outgoing = output['outgoing']

rel: Relation
for i, rel in enumerate(incoming + outgoing):
    print(f"{i+1}. {rel.node_1.name}, {rel.edge.name}, {rel.node_2.name}")