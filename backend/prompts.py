from langchain.prompts import ChatPromptTemplate

# Prompt used to extract relations from documents
EXTRACT_SYSTEM_PROMPT = """\
You are a network graph maker who extracts terms and their relations from a given context. 
You are provided with a context chunk (delimited by ```) Your task is to extract the ontology 
of terms mentioned in the given context. These terms should represent the key concepts as per the context. 
Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.
    Terms may include object, entity, location, organization, person, 
    condition, acronym, documents, service, concept, etc.
    Terms should be as atomistic and singular as possible

Thought 2: Think about how these terms can have one on one relation with other terms.
    Terms that are mentioned in the same sentence or the same paragraph are typically related to each other.
    Terms can be related to many other terms

Thought 3: Find out the relation between each such related pair of terms. 
    Format your output as a list of json. Each element of the list contains a pair of terms
    and the relation between them, like the follwing: 
    [
        {{
            "node_1": "A concept from extracted ontology",
            "node_2": "A related concept from extracted ontology",
            "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"
        }}, {{...}}
    ]\
"""

EXTRACT_HUMAN_PROMPT = "context: ```{input}```"

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXTRACT_SYSTEM_PROMPT),
    ("human", EXTRACT_HUMAN_PROMPT)
])

# Prompt used to filter the documents retrieved from the vectorstore
METADATA_SYSTEM_PROMPT = """\
You are tasked with converting a natural language query into a json query based to retrieve relevant documents from hybrid graph vector database.
Here are the fields you can use to filter the documents:
- doc_type: 'node' | 'edge' | 'raw' -> signifies whether you want to look up a node, an edge or the raw documents that were used to extract the relations from.

Depending on the query provided, thoughtfully decide what type of documents you want to filter.
node doc-types contain high level concepts, topics, entities, etc..
edge doc-types contain information about the relationships between nodes
raw contain the information that was used to extract the nodes and edges from
Only respond in JSON according to the guidelines provided. \
"""

METADATA_HUMAN_PROMPT = "query: ```{input}```"

METADATA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", METADATA_SYSTEM_PROMPT),
    ("human", METADATA_HUMAN_PROMPT)
])

# Summarizing all edges connected to a node
SUMMARIZE_SYSTEM_PROMPT = """\

"""

SUMMARIZE_HUMAN_PROMPT = """\

"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUMMARIZE_SYSTEM_PROMPT),
    ("human", SUMMARIZE_HUMAN_PROMPT)
])