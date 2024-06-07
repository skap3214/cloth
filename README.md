# 🕷️🕸️ Cloth

Better alternative to RAG combining knowledge graphs + vectorstores + databases.

## Quick Install

With pip

```bash
pip install cloth
```

## What is Cloth

**Cloth** is a library that improves the accuracy of your RAG application/pipeline using a combination of knowledge graphs, vectorstores and a PostgreSQL database.

Cloth uses Langchain data types and classes which makes it easy to integrate it into your existing Langchain powered application.

## Why Cloth

**Answer questions with high accuracy**

Retrieval Augmented Generation is great, but it cannot answer questions that require more complex types of retrieval. Many existing solutions require you to call the LLM multiple times sequentially during retrieval which can greatly slow down your application and increase costs. Cloth uses a combination of a vectorstore and a knowledge graph represented in a database to extract relations from your data to help answer all types of questions:

- high-level: Summarize x
- low-level: What is [insert specific topic]
- temporal: What happens during (time period)
- compare/contast: How is X and Y related?

**Multiple types of quick retrieval**

Cloth offers multiple different types of retrieval that can be optionally augmented with LLMs. Since your data is stored in a knowledge graph, it can extract/summarize high level topic with ease. Compare and contrasting entities can also be done by finding and summarizing paths betwen them in the knowledge graph. All the data in the knowledge graph can be found through exact keyword search and cosine similarity search using a vectorstore. Based on your type of retrieval, it also takes around the same time as a normal RAG application.

**Easy integration with Langchain**

Cloth uses multiple commonly used Langchain classes making it easy to integrate with your existing Langchain application. Specifically, Cloth classes use the Langchain Document(), Embeddings(), BaseChatModel(), BaseVectorstore() class, making it easy to swap them out with whichever one you want.

**Disadvantages**

- More space: Apart from data, relations are also stored in the vectorstore and a database.
- More time taken to add information.

## How

Cloth has only one class named `Graph` that you need to call to initialize, add, retrieve and visualize your RAG pipeline.

```python
from cloth import Neo4jGraphstore
from langchain.docstore.document import Document

graph = Neo4jGraphstore()

# Prepare your data
text = """
The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available."""

documents = [Document(page_content=text.strip())]

# Add
info = graph.add(documents)

# Retrieve information
nodes = graph.similarity_search("AI", doc_type="node") # Retrieve all similar nodes from knowledge graph
edges = graph.similarity_search("AI", doc_type="edge") # Retrieve all similar edges from knowledge graph
info = graph.similarity_search("AI", doc_type="raw") # Normal cosine similarity search of your documents
adj_edges = graph.get_adjacent_edges(query) # Retrieve all adjacent edges based on the most similar node of your query
adj_nodes = graph.get_adjacent_nodes(query) # Retrieve all adjacent nodes based on the most similar edge of your query
```
