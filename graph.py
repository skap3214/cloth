import os
from supabase import create_client
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from typing import List, Optional, Dict, Any, TypedDict, Literal
from langchain_core.output_parsers import JsonOutputParser
from prompts import EXTRACT_PROMPT, METADATA_PROMPT
from langchain.prompts import ChatPromptTemplate
from collections import deque
from pyvis.network import Network
import hashlib


class Relation(TypedDict):
    node_1: str
    edge: str
    node_2: str

class Graph:

    def __init__(
        self,
        collection_name: str = "graph",
        embeddings_model: Embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress=True),
        llm: BaseChatModel = ChatGroq(model="llama3-70b-8192", temperature=0.1),
        extract_prompt: ChatPromptTemplate = EXTRACT_PROMPT,
        metadata_prompt: ChatPromptTemplate = METADATA_PROMPT,
        persist_directory: str = "./local/vectorstore"
    ) -> None:
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.extract_prompt = extract_prompt
        self.metadata_prompt = metadata_prompt
        self.graph_chain = self.extract_prompt | self.llm | JsonOutputParser(pydantic_object=Relation)
        self.metadata_chain = self.metadata_prompt | self.llm | JsonOutputParser()
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings_model,
            persist_directory=self.persist_directory
        )

        self._database = create_client(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_KEY"),
        )

        self.nodes = self._database.table("nodes")
        self.edges = self._database.table("edges")

    def reset(self):
        self.vectorstore._client.reset()
        self.edges.delete()
        self.nodes.delete()
        return True

    def extract_relations(
        self, 
        documents: List[Document],
        k: Optional[int] = None
    ) -> List[List[Relation]]:
        batch_input = [{'input':doc.page_content} for doc in documents]
        # for document in documents:
            # relations = self.similarity_search(document.page_content, doc_type='edge', k=3)
        return self.graph_chain.batch(batch_input)

    def add(
        self, 
        documents: List[Document], 
        relations: Optional[List[List[Relation]]] = None,
        batch_size: int = 10,
    ):
        # Extract relations
        relations = relations or self.extract_relations(documents)

        # Add to graph
        self._add_to_vectorstore(
            documents=documents, 
            relations=relations
        )

        self._add_to_database(
            documents=documents,
            relations=relations,
            batch_size=batch_size
        )

    @staticmethod
    def generate_hash(input_string):

        return hashlib.sha256(input_string.encode()).hexdigest()

    def __insert_node(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        # Check if a node with the same data already exists
        response = self.nodes.select("id").eq("data", data).limit(1).execute()
        existing_node = None if len(response.data) == 0 else response.data[0]

        if existing_node:
            # Node already exists, return the existing node ID
            return existing_node["id"]
        else:
            # Insert a new node
            response = self.nodes.upsert({"data": data, "metadata": metadata}).execute()
            inserted_node = None if len(response.data) == 0 else response.data[0]

            return inserted_node["id"]

    def _insert_edge(
        self, 
        data: str, 
        from_node_data: str, 
        to_node_data: str, 
        page_content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        # Get the IDs of the source and target nodes
        from_node_id = self.__insert_node(from_node_data)
        to_node_id = self.__insert_node(to_node_data)

        # Check if an edge with the same data already exists
        response = self.edges.select("id").eq("data", data).execute()
        existing_edge = None if len(response.data) == 0 else response.data[0]

        if existing_edge:
            # Edge already exists, return the existing edge ID
            return existing_edge["id"]
        else:
            # Insert a new edge
            response = self.edges.upsert({
                "data": data,
                "source": from_node_id,
                "target": to_node_id,
                "page_content": page_content,
                "metadata": metadata
            }).execute()
            inserted_edge = None if len(response.data) == 0 else response.data[0]

            return inserted_edge["id"]

    def _add_to_vectorstore(
        self,
        documents: List[Document],
        relations: List[List[Relation]],
    ):
        nodes = set()
        edges = set()
        for relation in relations:
            for rel in relation:
                nodes.add(rel["node_1"])
                nodes.add(rel["node_2"])
                edges.add(rel['edge'])

        node_documents = [Document(page_content=node, metadata={'doc_type': 'node'}) for node in list(nodes)]
        edge_documents = [Document(page_content=edge, metadata={'doc_type': 'edge'}) for edge in list(edges)]
        [doc.metadata.update({'doc_type': 'raw'}) for doc in documents]
        all_documents = node_documents + edge_documents + documents

        ids = self.vectorstore.add_documents(
            documents=all_documents,
            ids=[self.generate_hash(doc.page_content) for doc in all_documents]
        )
        return ids

    def _add_to_database(
        self, 
        documents: List[Document], 
        relations: List[List[Relation]],
        batch_size: int = 10
    ):
        for i in range(0, len(documents), batch_size):
            batch_documents = documents[i:i + batch_size]
            batch_relations = relations[i:i + batch_size]

            for document, document_relations in zip(batch_documents, batch_relations):
                for relation in document_relations:
                    if not relation:
                        continue

                    self._insert_edge(
                        relation['edge'],
                        relation['node_1'],
                        relation['node_2'],
                        document.page_content,
                        document.metadata,
                    )

    def generate_filter(self, query: str):
        return self.metadata_chain.invoke({'input': query})

    def similarity_search(
        self, 
        query: str, 
        filter: Optional[Dict] = None,
        doc_type: Optional[Literal['raw', 'node', 'edge']] = None,
        k: int = 5,
        return_documents: bool = False,
        **kwargs
    ) -> List[Document] | List[str]:
        if filter and doc_type:
            filter.update({'doc_type':doc_type})
        else:
            filter = {'doc_type':doc_type}

        docs = self.vectorstore.similarity_search(
            query,
            k,
            filter,
            **kwargs
        )
        
        if return_documents: 
            return docs 
        else: 
            return [doc.page_content for doc in docs]

    def find_shortest_path(self, start_data: str, end_data: str) -> List[str]:
        """
        Find the shortest path between two nodes based on their data content.
        Args:
            start_data (str): The data content of the starting node.
            end_data (str): The data content of the ending node.

        Returns:
            List[str]: A list of node data representing the path from start to end,
                    or an empty list if no path exists.
        """
        # Similarity search to find 2 closest nodes
        start_data = self.similarity_search(start_data, doc_type='node', k=1)[0]
        end_data = self.similarity_search(end_data, doc_type='node', k=1)[0]
        
        # Get start and end node IDs
        start_node = self.nodes.select("id").eq("data", start_data).execute()
        end_node = self.nodes.select("id").eq("data", end_data).execute()

        if not start_node.data or not end_node.data:
            return []  # Start or end node does not exist

        start_node_id = start_node.data[0]['id']
        end_node_id = end_node.data[0]['id']

        # Initialize BFS
        queue = deque([start_node_id])
        paths = {start_node_id: [start_data]}
        visited = set([start_node_id])

        while queue:
            current_node_id = queue.popleft()

            # Retrieve all adjacent nodes
            inc = self.edges.select("target!inner(id), source!inner(id)").eq("target.id", current_node_id).execute()
            out = self.edges.select("target!inner(id), source!inner(id)").eq("source.id", current_node_id).execute()
            edges = inc.data + out.data

            for edge in edges:
                # Extracting scalar node ID values
                target_id = edge.get('target', {}).get('id')
                source_id = edge.get('source', {}).get('id')
                neighbor_node_id = target_id if source_id == current_node_id else source_id

                if neighbor_node_id not in visited:
                    visited.add(neighbor_node_id)
                    neighbor_data = self.nodes.select("data").eq("id", neighbor_node_id).execute()
                    if neighbor_data.data:
                        neighbor_data = neighbor_data.data[0]['data']
                        # Construct the path leading to this neighbor
                        paths[neighbor_node_id] = paths[current_node_id] + [neighbor_data]
                        if neighbor_node_id == end_node_id:
                            return paths[neighbor_node_id]  # Found the shortest path
                        queue.append(neighbor_node_id)

        return []  # No path found

    def get_adjacent_edges(self, node_data: str, edge_direction: str = 'both') -> List[Dict[str, Any]]:
        """
        Retrieve edges connected to a node based on the specified direction.
        
        Args:
            node_data (str): The data content of the node to lookup.
            edge_direction (str): Specifies the type of edges to retrieve:
                                'outgoing' for edges originating from this node,
                                'incoming' for edges directed to this node,
                                'both' for all edges connected to this node.

        Returns:
            List[Dict[str, Any]]: A list of edges with their metadata and connected node data.
        """
        # Get the node ID from the node data
        node = self.nodes.select("id").eq("data", node_data).single().execute().data
        if not node:
            return []  # Node does not exist

        node_id = node['id']
        edges = []

        # Fetch edges based on the specified direction
        if edge_direction == 'outgoing' or edge_direction == 'both':
            outgoing_edges = self.edges.select("id, target, data, metadata, page_content").eq("source", node_id).execute().data
            for edge in outgoing_edges:
                target_node_data = self.nodes.select("data").eq("id", edge['target']).single().execute().data['data']
                edges.append({
                    'type': 'outgoing',
                    'edge_id': edge['id'],
                    'node': target_node_data,
                    'relationship': edge['data'],
                    'metadata': edge['metadata'],
                    'page_content': edge['page_content']
                })

        if edge_direction == 'incoming' or edge_direction == 'both':
            incoming_edges = self.edges.select("id, source, data, metadata, page_content").eq("target", node_id).execute().data
            for edge in incoming_edges:
                source_node_data = self.nodes.select("data").eq("id", edge['source']).single().execute().data['data']
                edges.append({
                    'type': 'incoming',
                    'edge_id': edge['id'],
                    'node': source_node_data,
                    'relationship': edge['data'],
                    'metadata': edge['metadata'],
                    'page_content': edge['page_content']
                })

        return edges
    
    def _get_edges(self):
        edges = self.edges.select("*").execute().data
        return edges
    
    def _get_nodes(self):
        nodes = self.nodes.select("*").execute().data
        return nodes
    

    def visualize(self, relations: List[List[Relation]]):
        edges = self._get_edges()
        nodes = self._get_nodes()




if __name__ == "__main__":
    graph = Graph()
    node_1 = "agents"
    node_2 = "bitter lesson"
    path = graph.find_shortest_path(node_1, node_2)
    print(path)