import os
from time import time
from supabase import create_client
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from typing import List, Optional, Dict, Any, Literal
from langchain_core.output_parsers import JsonOutputParser
from .prompts import EXTRACT_PROMPT, METADATA_PROMPT
from langchain.prompts import ChatPromptTemplate
from pyvis.network import Network
import hashlib
from .utils.logger import get_logger
from .types import Relation
logger = get_logger()


class Graphstore:

    def __init__(
        self,
        collection_name: str = "graph",
        embeddings_model: Embeddings = OllamaEmbeddings(model="nomic-embed-text"), # OpenAIEmbeddings(model="text-embedding-3-small")
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
        start = time()
        batch_input = [{'input':doc.page_content} for doc in documents]
        # for document in documents:
            # relations = self.similarity_search(document.page_content, doc_type='edge', k=3)
        batch_response = self.graph_chain.batch(batch_input)
        logger.debug(f"{len(documents)} Relations generated in {time() - start}")
        return batch_response

    def add(
        self, 
        documents: List[Document], 
        relations: Optional[List[List[Relation]]] = None,
        batch_size: int = 10,
    ):
        # Extract relations
        relations = relations or self.extract_relations(documents)
        relations = self.preprocess_graph_data(relations)

        # Add to graph
        start = time()
        self._add_to_vectorstore(
            documents=documents, 
            relations=relations
        )

        self._add_to_database(
            documents=documents,
            relations=relations,
            batch_size=batch_size
        )
        logger.debug(f"Documents and Relations added in {time() - start}")

        return {
            'documents': documents,
            'relations': relations
        }

    @staticmethod
    def preprocess_graph_data(relations: List[List[Relation]]) -> List[List[Relation]]:
        # Collect all nodes and edges
        all_nodes = set()
        connected_nodes = set()
        edges = []

        for relation_list in relations:
            for relation in relation_list:
                node_1 = relation['node_1']
                node_2 = relation['node_2']
                edge = relation['edge']
                
                all_nodes.add(node_1)
                all_nodes.add(node_2)
                connected_nodes.add(node_1)
                connected_nodes.add(node_2)
                edges.append((node_1, node_2, edge))
        
        # Identify isolated nodes
        isolated_nodes = all_nodes - connected_nodes
        logger.debug(f"{isolated_nodes=}")

        # Remove isolated nodes from the edges list
        processed_data = []
        for relation_list in relations:
            new_relation_list = []
            for relation in relation_list:
                if relation['node_1'] not in isolated_nodes and relation['node_2'] not in isolated_nodes:
                    new_relation_list.append(relation)
            if new_relation_list:
                processed_data.append(new_relation_list)
        
        return processed_data

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

    def find_paths(self, start_node: str, end_node: str, max_depth: int = 3) -> List[List[str]]:
        start_node_id = self.__get_node_id(start_node)
        end_node_id = self.__get_node_id(end_node)
        if not start_node_id or not end_node_id:
            return []

        paths = []
        stack = [(start_node_id, [start_node_id])]

        while stack:
            (current_node, path) = stack.pop()
            if current_node == end_node_id:
                paths.append(path)
            elif len(path) < max_depth:
                adjacent_nodes = self.__get_adjacent_nodes(current_node)
                for next_node in adjacent_nodes:
                    if next_node not in path:
                        stack.append((next_node, path + [next_node]))

        return [[self.__get_node_data(node_id) for node_id in path] for path in paths]

    def __get_node_id(self, node_data: str) -> Optional[str]:
        node = self.nodes.select("id").eq("data", node_data).single().execute().data
        return node['id'] if node else None

    def __get_node_data(self, node_id: str) -> Optional[str]:
        node = self.nodes.select("data").eq("id", node_id).single().execute().data
        return node['data'] if node else None

    def __get_adjacent_nodes(self, node_id: str) -> List[str]:
        outgoing_edges = self.edges.select("target").eq("source", node_id).execute().data
        incoming_edges = self.edges.select("source").eq("target", node_id).execute().data
        return [edge['target'] for edge in outgoing_edges] + [edge['source'] for edge in incoming_edges]

    def visualize(
        self, 
        relations: Optional[List[List[Relation]]] = None,
        **network_kwargs
    ):
        
        net = Network(
            directed=True,
            **network_kwargs
        )
        if not relations:
            nodes = self.nodes.select("*").execute().data
            edges = self.edges.select("*").execute().data
            for node in nodes:
                net.add_node(node['id'], label=node['data'])
            for edge in edges:
                net.add_edge(edge['source'], edge['target'], title=edge['data'])

            net.write_html(name='index.html')
        else:
            nodes = []
            edges = []
            for rel_list in relations:
                for rel in rel_list:
                    node_1 = rel.node_1
                    node_1_id = self.generate_hash(node_1)
                    edge = rel.edge
                    node_2 = rel.node_2
                    node_2_id = self.generate_hash(node_2)
                    net.add_node(node_1_id, label=node_1)
                    net.add_node(node_2_id, label=node_2)
                    net.add_edge(node_1_id, node_2_id, title=edge)


if __name__ == "__main__":
    from langchain.docstore.document import Document
    from textwrap import dedent
    text = dedent("""
    The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.
    >>
    In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that ``brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.
    >>
    A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.
    >>
    In speech recognition, there was an early competition, sponsored by DARPA, in the 1970s. Entrants included a host of special methods that took advantage of human knowledge---knowledge of words, of phonemes, of the human vocal tract, etc. On the other side were newer methods that were more statistical in nature and did much more computation, based on hidden Markov models (HMMs). Again, the statistical methods won out over the human-knowledge-based methods. This led to a major change in all of natural language processing, gradually over decades, where statistics and computation came to dominate the field. The recent rise of deep learning in speech recognition is the most recent step in this consistent direction. Deep learning methods rely even less on human knowledge, and use even more computation, together with learning on huge training sets, to produce dramatically better speech recognition systems. As in the games, researchers always tried to make systems that worked the way the researchers thought their own minds worked---they tried to put that knowledge in their systems---but it proved ultimately counterproductive, and a colossal waste of researcher's time, when, through Moore's law, massive computation became available and a means was found to put it to good use.
    >>
    In computer vision, there has been a similar pattern. Early methods conceived of vision as searching for edges, or generalized cylinders, or in terms of SIFT features. But today all this is discarded. Modern deep-learning neural networks use only the notions of convolution and certain kinds of invariances, and perform much better.
    >>
    This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that 1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning. The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.
    >>
    One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.
    >>
    The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.\
    """)

    documents = [Document(page_content=ex.strip()) for ex in text.split(">>")]
    graph = Graphstore()

    # graph.add(documents)