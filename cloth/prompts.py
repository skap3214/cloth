from langchain.prompts import ChatPromptTemplate


# Prompt used to extract relations from documents
EXTRACT_FORMAT_PROMPT = """\
[
    {{
        "node_1": {{
            "name": "Concept from extracted ontology",
        }},
        "edge": {{
            "name": "relationship between the two concepts, node_1 and node_2 in one or two sentences",
        }},
        "node_2": {{
            "name": "A related concept from extracted ontology",
        }}
    }}, {{...}}
]
"""
EXTRACT_SYSTEM_PROMPT = """\
You are a network graph maker who extracts terms and their relations from a given context. 
You are provided with a context chunk (delimited by ```) Your task is to extract the ontology 
of terms mentioned in the given context. These terms should represent the key entities as per the context.
You also need to find out the relation between each such related pair of terms.
The type of terms you need to extract are listed below:
{node_type}
Be as atomistic and singular as possible.
Think about how these terms can have one on one relation with other terms.
Terms that are mentioned in the same sentence or the same paragraph are typically related to each other.
The type of relations you need to extract are listed below:
{edge_type}
Format your output as a list of json. Each element of the list contains a pair of terms
and the relation between them, like the following:
""" + EXTRACT_FORMAT_PROMPT + """
Make sure the strings are in double quotes, escape special characters when needed, \
all nodes and edge names should be title cased, \
and your response should start and end with '[' and ']' \
"""

EXTRACT_HUMAN_PROMPT = "context: ```{input}``` ONLY output the valid  JSON object. Start your response with ["

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