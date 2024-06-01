import os
from time import time
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from typing import List, Optional, Dict, Any, Literal, Generator
from langchain_core.output_parsers import JsonOutputParser
from .prompts import EXTRACT_PROMPT, METADATA_PROMPT
from langchain.prompts import ChatPromptTemplate
from .utils.logger import get_logger
from .utils.id import generate_id
from .types import Relation, Node, Edge
from neo4j import GraphDatabase, Session

logger = get_logger()

class Neo4jGraphstore:

    def __init__(
        self,
        collection_name: str = "graph",
        embeddings_model: Embeddings = OllamaEmbeddings(model="nomic-embed-text"), # OpenAIEmbeddings(model="text-embedding-3-small")
        llm: BaseChatModel = ChatGroq(model="llama3-70b-8192", temperature=0.1),
        extract_prompt: ChatPromptTemplate = EXTRACT_PROMPT,
        node_type: Optional[List[str] | str] = None,
        edge_type: Optional[List[str] | str] = None,
        metadata_prompt: ChatPromptTemplate = METADATA_PROMPT,
        persist_directory: str = "./local/vectorstore",
        neo4j_uri: str = os.getenv("NEO4J_URI"),
        neo4j_user: str = os.getenv("NEO4J_USER"),
        neo4j_password: str = os.getenv("NEO4J_PASSWORD")
    ) -> None:
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.node_type = node_type or [
                "object", "entity", "location", "organization", 
                "person", "condition", "acronym", "documents", 
                "service", "concept", "emotion", "etc..."
        ]
        self.edge_type = edge_type or "(No Restrictions)"
        self.extract_prompt = extract_prompt.partial(node_type=str(self.node_type), edge_type=self.edge_type)
        self.metadata_prompt = metadata_prompt
        self.graph_chain = self.extract_prompt | self.llm | JsonOutputParser(pydantic_object=Relation)
        self.metadata_chain = self.metadata_prompt | self.llm | JsonOutputParser()

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings_model,
            persist_directory=self.persist_directory,
        )
        self.vectorstore._client_settings.allow_reset = True

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def reset(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        self.vectorstore._client.reset()
        return True

    def extract_relations(self, documents: List[Document], stream: bool = False) -> List[List[Relation]]:
        if not stream:
            start = time()
            batch_input = [{'input': doc.page_content} for doc in documents]
            batch_response: List[List[Dict]] = self.graph_chain.batch(batch_input)
            logger.debug(f"{len(documents)} Relations generated in {time() - start}")
            formatted_batch_response = [[Relation(**relation) for relation in response] for response in batch_response]
            return formatted_batch_response
        else:
            return self._extract_relation_streaming(documents)

    def _extract_relation_streaming(self, documents: List[Document]) -> Generator[List[Relation], Any, None]:
        start = time()
        for doc in documents:
            relation_list = self.graph_chain.invoke({'input': doc.page_content})
            yield [Relation(**rel) for rel in relation_list]
        logger.debug(f"{len(documents)} Relations generated in {time() - start}")

    def add(self, documents: List[Document], relations: Optional[List[List[Relation]]] = None, stream: bool = False):
        if stream:
            return self._add_streaming(documents=documents, relations=relations)
        relations = relations or self.extract_relations(documents)
        start = time()
        self._add_to_vectorstore(documents=documents, relations=relations)
        self._add_to_database(documents=documents, relations=relations)
        logger.debug(f"Documents and Relations added in {time() - start}")
        return {'documents': documents, 'relations': relations}

    def _add_streaming(self, documents: List[Document], relations: Optional[List[List[Relation]]] = None):
        rel_stream = relations or self.extract_relations(documents, stream=True)
        final_relations: List[List[Relation]] = []
        for rel_list, doc in zip(rel_stream, documents):
            yield {'document': doc, 'relations': rel_list}
            final_relations.append(rel_list)
        self._add_to_vectorstore(documents=documents, relations=final_relations)
        self._add_to_database(documents=documents, relations=final_relations)

    def _add_to_vectorstore(self, documents: List[Document], relations: List[List[Relation]]):
        nodes = set()
        edges = list()
        for relation in relations:
            for rel in relation:
                rel = rel.model_dump()
                nodes.add(rel["node_1"]['name'])
                nodes.add(rel["node_2"]['name'])
                edges.append(rel['edge']['name'])
        node_documents = [Document(page_content=node, metadata={'doc_type': 'node'}) for node in list(nodes)]
        edge_documents = [Document(page_content=edge, metadata={'doc_type': 'edge'}) for edge in edges]
        [doc.metadata.update({'doc_type': 'raw'}) for doc in documents]
        all_documents = node_documents + edge_documents + documents
        ids = self.vectorstore.add_documents(
            documents=all_documents,
            ids=[
                generate_id(doc.page_content) if doc.metadata.get('doc_type') != 'edge' else generate_id(always_unique=True) 
                for doc in all_documents
            ]
        )
        return ids

    def _add_to_database(self, documents: List[Document], relations: List[List[Relation]]):
            document_ids = [generate_id(doc.page_content) for doc in documents]
            with self.driver.session() as session:
                for document, document_relations, document_id in zip(documents, relations, document_ids):
                    self.insert_relation(session, document, document_relations, document_id)

    def insert_relation(self, session: Session, document: Document, relations: List[Relation], document_id: Optional[str] = None):
        if not document_id:
            document_id = self.generate_hash(document.page_content)

        metadata = document.metadata
        metadata['page_content'] = document.page_content
        metadata['type'] = document.type
        metadata['id'] = document_id

        create_relations_query = """
        MERGE (d:Document {id: $doc_id})
        SET d += $metadata
        WITH d
        UNWIND $relations AS rel
        MERGE (n1:Node {name: rel.node_1.name, id: rel.node_1.id})
        MERGE (n2:Node {name: rel.node_2.name, id: rel.node_2.id})
        MERGE (n1)-[:Edge {name: rel.edge.name, document_id: $doc_id}]->(n2)
        MERGE (n1)-[:From {document_id: $doc_id}]->(d)
        MERGE (n2)-[:From {document_id: $doc_id}]->(d)
        """

        session.run(create_relations_query, doc_id=document_id, metadata=metadata, relations=[rel.model_dump() for rel in relations])

    def generate_filter(self, query: str):
        return self.metadata_chain.invoke({'input': query})

    def similarity_search(self, query: str, filter: Optional[Dict] = None, doc_type: Optional[Literal['raw', 'node', 'edge']] = None, k: int = 5, return_documents: bool = False, **kwargs) -> List[Document] | List[str]:
        if not doc_type:
            doc_type = "raw"
        if filter and doc_type:
            filter.update({'doc_type': doc_type})
        else:
            filter = {'doc_type': doc_type}
        docs = self.vectorstore.similarity_search(query, k, filter, **kwargs)
        if return_documents:
            return docs
        else:
            return [doc.page_content for doc in docs]

    def get_adjacent_edges(self, node_name: str, edge_direction: str = 'both') -> Dict[str, List[Dict[str, Any]]]:
        with self.neo4j_driver.session() as session:
            edges = {'outgoing': [], 'incoming': []}

            if edge_direction in ['both', 'outgoing']:
                result = session.run(
                    """
                    MATCH (n {name: $node_name})-[r]->(m)
                    WHERE NOT m:Document
                    RETURN r, m.name AS target_node_name, labels(m) AS target_labels
                    """, node_name=node_name
                )
                for record in result:
                    edges['outgoing'].append({
                        'start': node_name,
                        'end': record['target_node_name'],
                        'edge': Edge(id=record['r'].id, name=record['r'].type, type=record['r'].type),
                        'target_node': Node(id=record['r'].end_node.id, name=record['target_node_name'], type=record['target_labels'][0])
                    })

            if edge_direction in ['both', 'incoming']:
                result = session.run(
                    """
                    MATCH (n {name: $node_name})<-[r]-(m)
                    WHERE NOT m:Document
                    RETURN r, m.name AS source_node_name, labels(m) AS source_labels
                    """, node_name=node_name
                )
                for record in result:
                    edges['incoming'].append({
                        'start': record['source_node_name'],
                        'end': node_name,
                        'edge': Edge(id=record['r'].id, name=record['r'].type, type=record['r'].type),
                        'source_node': Node(id=record['r'].start_node.id, name=record['source_node_name'], type=record['source_labels'][0])
                    })

            return edges

    def find_paths(self, start_node: str, end_node: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH p=shortestPath((start {{name: "{start_node}"}})-[*..{max_depth}]-(end {{name: "{end_node}"}}))
                WHERE NONE(n IN nodes(p) WHERE n:Document)
                RETURN nodes(p) AS nodes, relationships(p) AS relationships
                """, start_node=start_node, end_node=end_node, max_depth=max_depth
            )
            paths = []
            for record in result:
                nodes = [{'name': node['name'], 'labels': list(node.labels)} for node in record['nodes']]
                relationships = [{'type': rel.type, 'start_node': rel.start_node['name'], 'end_node': rel.end_node['name'], 'properties': dict(rel)} for rel in record['relationships']]
                path = {'start_node': nodes[0]['name'], 'end_node': nodes[-1]['name'], 'nodes': nodes, 'relationships': relationships}
                paths.append(path)
            return paths

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
    graph = Neo4jGraphstore()
    graph.reset()
    graph.add(documents)
