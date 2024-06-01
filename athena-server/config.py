from cloth.neo4j_graph import Neo4jGraphstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class Config:
    GRAPH = Neo4jGraphstore(collection_name="graph")

    SPLITTER = RecursiveCharacterTextSplitter(
        keep_separator=False,
        chunk_size=500,
        chunk_overlap=30
    )
    LLM = ChatGroq(model="llama3-70b-8192", temperature=0.1)

    SYSTEM_PROMPT = """\
You are a helpful assistant. You are tasked with answering questions about a retrieved context given to you in delimiters.
```
{context}
```
If you don't know the answer, just say you don't know.\
"""

    HUMAN_PROMPT = """{input}"""

    PROMPT = ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        ('human', HUMAN_PROMPT)
    ])

    CHAT_CHAIN = PROMPT | LLM |  StrOutputParser()