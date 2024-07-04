import os
from time import time
from typing import List, Optional, Dict, Any, Literal, Generator, Set

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from neo4j import GraphDatabase, Session

from cloth.prompts.graph import EXTRACT_PROMPT, METADATA_PROMPT
from cloth.utils.logger import get_logger
from cloth.utils.id import generate_id
from cloth.types.graph import Relation, Node, Edge

logger = get_logger()

class Neo4jVectorGraphstore:

    def __init__(
        self,
        collection_name: str = "graph",
        embeddings_model: Embeddings = OllamaEmbeddings(model="nomic-embed-text"), # OpenAIEmbeddings(model="text-embedding-3-small")
        llm: BaseChatModel = ChatGroq(model="llama3-70b-8192", temperature=0.1),
        output_fixing_llm: BaseChatModel = ChatGroq(model="llama3-8b-8192", temperature=0.1),
        extract_prompt: ChatPromptTemplate = EXTRACT_PROMPT,
        node_type: Optional[List[str] | str] = None,
        edge_type: Optional[List[str] | str] = None,
        metadata_prompt: ChatPromptTemplate = METADATA_PROMPT,
        vectorstore: Optional[VectorStore] = None,
        neo4j_uri: str = os.getenv("NEO4J_URI"),
        neo4j_user: str = os.getenv("NEO4J_USER"),
        neo4j_password: str = os.getenv("NEO4J_PASSWORD")
    ) -> None:
        """Create a Graphstore using Neo4j

        Args:
            collection_name (str, optional): Name of the collection. Used to create the vectorstore. Defaults to "graph".
            embeddings_model (Embeddings, optional): embeddings model to use. Uses Langchain's Embeddings class. Defaults to OllamaEmbeddings(model="nomic-embed-text").
            extract_prompt (ChatPromptTemplate, optional): prompt used to extract relations from a given document. Uses Langchain's ChatPrompt Template to define the prompt. Defaults to EXTRACT_PROMPT.
            node_type (Optional[List[str]  |  str], optional): Type of nodes to extract from the documents. Defaults to all nodes.
            edge_type (Optional[List[str]  |  str], optional): Type of edges/relations/links to extract from the documents. Defaults to all edges.
            metadata_prompt (ChatPromptTemplate, optional): Metadata prompt to filter searches using an LLM. Defaults to METADATA_PROMPT.
            persist_directory (str, optional): Directory to persist your vectorstore. Defaults to "./local/vectorstore".
            neo4j_uri (str, optional): Neo4j URI. Defaults to os.getenv("NEO4J_URI").
            neo4j_user (str, optional): Neo4j User. Defaults to os.getenv("NEO4J_USER").
            neo4j_password (str, optional): Neo4j password. Defaults to os.getenv("NEO4J_PASSWORD").
        """
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
        self.metadata_chain = self.metadata_prompt | self.llm | OutputFixingParser.from_llm(llm=output_fixing_llm, parser=JsonOutputParser())

        self.collection_name = collection_name
        self.vectorstore = vectorstore or PineconeVectorStore(
            embedding=self.embeddings_model,
            index_name=self.collection_name
        )
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


    def extract_relations(
        self, 
        documents: List[Document], 
        stream: bool = False, 
        metadata: Optional[Dict] = None
    ) -> List[List[Relation]]:

        if not metadata:
            metadata = {}
        if not stream:
            start = time()
            batch_input = [{'input': doc.page_content} for doc in documents]
            batch_response: List[List[Dict]] = self.graph_chain.batch(batch_input)
            logger.debug(f"{len(documents)} Relations generated in {time() - start}")
            formatted_batch_response = []
            for response in batch_response:
                relations = []
                if not response:
                    continue
                for relation in response:
                    if not relation:
                        continue
                    # Add metadata into relations before adding it to the pydantic object
                    relation['node_1'].setdefault('metadata', {}).update(metadata)
                    relation['node_2'].setdefault('metadata', {}).update(metadata)
                    relation['edge'].setdefault('metadata', {}).update(metadata)
                    relations.append(Relation(**relation))
                formatted_batch_response.append(relations)
            return formatted_batch_response
        else:
            return self._extract_relation_streaming(documents, metadata=metadata)


    def _extract_relation_streaming(
        self, 
        documents: List[Document],
        metadata: Optional[Dict] = None
    ) -> Generator[List[Relation], Any, None]:
        start = time()
        for doc in documents:
            relation_list = self.graph_chain.invoke({'input': doc.page_content})
            if relation_list:
                yield_list = []
                for rel in relation_list:
                    rel['node_1'].setdefault('metadata', {}).update(metadata)
                    rel['node_2'].setdefault('metadata', {}).update(metadata)
                    rel['edge'].setdefault('metadata', {}).update(metadata)
                    yield_list.append(Relation(**rel))
                yield yield_list
        logger.debug(f"{len(documents)} Relations generated in {time() - start}")


    def add(
            self, 
            documents: List[Document], 
            relations: Optional[List[List[Relation]]] = None, 
            stream: bool = False, 
            metadata: Optional[Dict] = None
        ):
        if stream:
            return self._add_streaming(documents=documents, relations=relations, metadata=metadata)
        relations = relations or self.extract_relations(documents, metadata=metadata)
        start = time()
        self._add_to_vectorstore(documents=documents, relations=relations, metadata=metadata)
        self._add_to_database(documents=documents, relations=relations, metadata=metadata)
        logger.debug(f"Documents and Relations added in {time() - start}")
        return {'documents': documents, 'relations': relations}


    def _add_streaming(
            self, 
            documents: List[Document], 
            relations: Optional[List[List[Relation]]] = None, 
            metadata: Optional[Dict] = None
        ):
        rel_stream = relations or self.extract_relations(documents, metadata=metadata, stream=True)
        final_relations: List[List[Relation]] = []
        for rel_list, doc in zip(rel_stream, documents):
            yield {'document': doc, 'relations': rel_list}
            final_relations.append(rel_list)
        self._add_to_vectorstore(documents=documents, relations=final_relations, metadata=metadata)
        self._add_to_database(documents=documents, relations=final_relations, metadata=metadata)


    def _add_to_vectorstore(
            self, 
            documents: List[Document], 
            relations: List[List[Relation]],
            metadata: Optional[Dict] = None
        ):
        if not metadata:
            metadata = {}
        node_set: Set[str] = set()
        node_documents: List[Document] = []
        node_ids: List[str] = []

        edge_set: Set[str] = set()
        edge_documents: List[Document] = []
        edge_ids: List[str] = []

        document_set: Set[str] = set()
        raw_documents: List[Document] = []
        raw_ids: List[str] = [] # Contains page_content as id

        for relation, document in zip(relations, documents):
            for rel in relation:
                if (node_id:= rel.node_1.id) not in node_set:
                    node_documents.append(
                        Document(
                            page_content=rel.node_1.name, 
                            metadata={
                                'doc_type': 'node',
                                **metadata
                            }
                        )
                    )
                    node_ids.append(node_id)
                    node_set.add(node_id)

                if (node_id:= rel.node_2.id) not in node_set:
                    node_documents.append(
                        Document(
                            page_content=rel.node_2.name, 
                            metadata={
                                'doc_type': 'node',
                                **metadata
                            }
                        )
                    )
                    node_ids.append(node_id)
                    node_set.add(node_id)

                if (edge_id:= rel.edge.id) not in edge_set:
                    edge_documents.append(
                        Document(
                            page_content=rel.edge.name, 
                            metadata={
                                'doc_type': 'edge',
                                'source_node': rel.node_1.id,
                                'target_node': rel.node_2.id,
                                **metadata
                            }
                        )
                    )
                    #TODO: decide id based on (node, edge, node, metadata) or (node, edge, node, document, metadata)?
                    edge_ids.append(edge_id)
                    edge_set.add(edge_id)

            if (page_content:= document.page_content) not in document_set:
                raw_documents.append(
                    Document(
                        page_content=document.page_content, 
                        metadata={
                            'doc_type': 'raw',
                            **metadata
                        }
                    )
                )
                d_id = generate_id(page_content=document.page_content, **metadata)
                raw_ids.append(d_id)
                document_set.add(page_content)

        ids = self.vectorstore.add_documents(
            documents=node_documents + edge_documents + raw_documents,
            ids=node_ids + edge_ids + raw_ids
        )
        return ids


    def _add_to_database(
            self, 
            documents: List[Document], 
            relations: List[List[Relation]],
            metadata: Optional[Dict] = None
        ):
            metadata = metadata or {}
            document_ids = [generate_id(**{**metadata, 'page_content': doc.page_content}) for doc in documents]
            with self.driver.session() as session:
                for document, document_relations, document_id in zip(documents, relations, document_ids):
                    self.insert_relation(session, document, document_relations, document_id, metadata)


    def insert_relation(
            self, 
            session: Session, 
            document: Document, 
            relations: List[Relation], 
            document_id: str,
            graph_metadata: Optional[Dict[str, Any]] = None
        ):
        if not graph_metadata:
            graph_metadata = {}


        metadata = document.metadata
        metadata['page_content'] = document.page_content
        metadata['type'] = document.type
        metadata['id'] = document_id

        # Combine document metadata with graph metadata
        combined_metadata = {**metadata, **graph_metadata}

        # Generate the SET clauses
        document_set_clause = ', '.join([f"d.{key} = $combined_metadata.{key}" for key in combined_metadata.keys()])
        node_set_clause = ', '.join([f"n1.{key} = $graph_metadata.{key}" for key in graph_metadata.keys()])
        edge_set_clause = ', '.join([f"r.{key} = $graph_metadata.{key}" for key in graph_metadata.keys()])
        from_set_clause1 = ', '.join([f"f1.{key} = $graph_metadata.{key}" for key in graph_metadata.keys()])
        from_set_clause2 = ', '.join([f"f2.{key} = $graph_metadata.{key}" for key in graph_metadata.keys()])

        # Helper function to format SET clauses conditionally
        def format_set_clause(clause):
            return f"ON CREATE SET {clause}" if clause else ""

        node_set_clause = format_set_clause(node_set_clause)
        edge_set_clause = format_set_clause(edge_set_clause)
        from_set_clause1 = format_set_clause(from_set_clause1)
        from_set_clause2 = format_set_clause(from_set_clause2)
        graph_metadata_formatted = f", {', '.join([f'{key}: $graph_metadata.{key}' for key in graph_metadata.keys()])}" if graph_metadata else ""

        create_relations_query = f"""
        MERGE (d:Document {{id: $doc_id}})
        SET {document_set_clause}
        WITH d
        UNWIND $relations AS rel
        MERGE (n1:Node {{name: rel.node_1.name, id: rel.node_1.id{graph_metadata_formatted}}})
        {node_set_clause}
        MERGE (n2:Node {{name: rel.node_2.name, id: rel.node_2.id{graph_metadata_formatted}}})
        {node_set_clause.replace("n1", "n2")}
        MERGE (n1)-[r:Edge {{name: rel.edge.name, document_id: $doc_id, id: rel.edge.id{graph_metadata_formatted}}}]->(n2)
        {edge_set_clause}
        MERGE (n1)-[f1:From {{document_id: $doc_id{graph_metadata_formatted}}}]->(d)
        {from_set_clause1}
        MERGE (n2)-[f2:From {{document_id: $doc_id{graph_metadata_formatted}}}]->(d)
        {from_set_clause2}
        """

        session.run(create_relations_query, doc_id=document_id, combined_metadata=combined_metadata, graph_metadata=graph_metadata, relations=[rel.model_dump() for rel in relations])


    def generate_filter(
            self, 
            query: str
    ):
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
        if not doc_type:
            doc_type = "raw"
        if filter and doc_type:
            filter.update({'doc_type': doc_type})
        else:
            filter = {'doc_type': doc_type}
            docs = self.vectorstore.similarity_search(
            query=query, 
            k=k, 
            filter=filter, 
            **kwargs
        )
        if return_documents:
            return docs
        else:
            return [doc.page_content for doc in docs]


    def get_adjacent_edges(
            self, 
            node_name: str, 
            edge_direction: Literal['both', 'incoming', 'outgoing'] = 'both',
            graph_metadata: Optional[Dict] = None
    ) -> Dict[str, List[Relation]]:
        """records, incoming, outgoing"""
        if graph_metadata:
            graph_metadata.pop("doc_type", None)
            n_metadata_conditions = ' AND '.join([f"n.{key} = $metadata.{key}" for key in graph_metadata.keys()])
            m_metadata_conditions = ' AND '.join([f"m.{key} = $metadata.{key}" for key in graph_metadata.keys()])
        else:
            n_metadata_conditions = ""
            m_metadata_conditions = ""

        # Helper function to format WHERE clause conditionally
        def format_where_clause(n_conditions, m_conditions):
            if n_conditions and m_conditions:
                return f"WHERE {n_conditions} AND {m_conditions} AND NOT m:Document"
            elif n_conditions:
                return f"WHERE {n_conditions} AND NOT m:Document"
            elif m_conditions:
                return f"WHERE {m_conditions} AND NOT m:Document"
            else:
                return "WHERE NOT m:Document"

        with self.driver.session() as session:
            edges = {'outgoing': [], 'incoming': [], 'records': []}
            if edge_direction in ['both', 'outgoing']:
                result = session.run(
                    f"""
                    MATCH (n {{name: $node_name}})-[r]->(m)
                    {format_where_clause(n_metadata_conditions, m_metadata_conditions)}
                    RETURN n as source, r as edge, m as target
                    """, node_name=node_name, metadata=graph_metadata
                )
                for record in result:
                    edges['records'].append(record)
                    source_metadata = dict(record['source'])
                    edge_metadata = dict(record['edge'])
                    target_metadata = dict(record['target'])
                    node_1 = Node(
                        id=source_metadata.pop('id'),
                        name=source_metadata.pop('name'), 
                        metadata=source_metadata
                    )
                    edge = Edge(
                        id=edge_metadata.pop('id'),
                        name=edge_metadata.pop('name'), 
                        metadata=edge_metadata
                    )
                    node_2 = Node(
                        id=target_metadata.pop('id'),
                        name=target_metadata.pop('name'), 
                        metadata=target_metadata
                    )
                    relation = Relation(
                        node_1=node_1,
                        edge=edge,
                        node_2=node_2,
                    )
                    edges['outgoing'].append(relation)

            if edge_direction in ['both', 'incoming']:
                result = session.run(
                    f"""
                    MATCH (n {{name: $node_name}})<-[r]-(m)
                    {format_where_clause(n_metadata_conditions, m_metadata_conditions)}
                    RETURN n as target, r as edge, m as source
                    """, node_name=node_name, metadata=graph_metadata
                )
                for record in result:
                    edges['records'].append(record)
                    source_metadata = dict(record['source'])
                    edge_metadata = dict(record['edge'])
                    target_metadata = dict(record['target'])
                    node_1 = Node(
                        id=source_metadata.pop('id'),
                        name=source_metadata.pop('name'), 
                        metadata=source_metadata
                    )
                    edge = Edge(
                        id=edge_metadata.pop('id'),
                        name=edge_metadata.pop('name'), 
                        metadata=edge_metadata
                    )
                    node_2 = Node(
                        id=target_metadata.pop('id'),
                        name=target_metadata.pop('name'), 
                        metadata=target_metadata
                    )
                    relation = Relation(
                        node_1=node_1,
                        edge=edge,
                        node_2=node_2
                    )
                    edges['incoming'].append(relation)

            return edges


    def get_adjacent_nodes(
            self, 
            edge_name: str,
            graph_metadata: Optional[Dict] = None
    ) -> Dict[str, List[Relation]]:
        if not graph_metadata:
            graph_metadata = {}
        if graph_metadata:
            graph_metadata.pop("doc_type", None)
        n_metadata_conditions = ' AND '.join([f"n.{key} = $metadata.{key}" for key in graph_metadata.keys()])
        m_metadata_conditions = ' AND '.join([f"m.{key} = $metadata.{key}" for key in graph_metadata.keys()])

        # Helper function to format WHERE clause conditionally
        def format_where_clause(n_conditions, m_conditions):
            if n_conditions and m_conditions:
                return f"WHERE {n_conditions} AND {m_conditions} AND NOT m:Document"
            elif n_conditions:
                return f"WHERE {n_conditions} AND NOT m:Document"
            elif m_conditions:
                return f"WHERE {m_conditions} AND NOT m:Document"
            else:
                return "WHERE NOT m:Document"

        with self.driver.session() as session:
            edges = {'relations': [], 'records': []}

            result = session.run(
                f"""
                MATCH (n)-[r:Edge {{name: $edge_name}}]->(m)
                {format_where_clause(n_metadata_conditions, m_metadata_conditions)}
                RETURN n as source, r as edge, m as target
                """, edge_name=edge_name, metadata=graph_metadata
            )
            for record in result:
                edges['records'].append(record)
                source_metadata = dict(record['source'])
                edge_metadata = dict(record['edge'])
                target_metadata = dict(record['target'])
                node_1 = Node(
                    id=source_metadata.pop('id'),
                    name=source_metadata.pop('name'), 
                    metadata=source_metadata
                )
                edge = Edge(
                    id=edge_metadata.pop('id'),
                    name=edge_metadata.pop('name'), 
                    metadata=edge_metadata
                )
                node_2 = Node(
                    id=target_metadata.pop('id'),
                    name=target_metadata.pop('name'), 
                    metadata=target_metadata
                )
                relation = Relation(
                    node_1=node_1,
                    edge=edge,
                    node_2=node_2,
                )
                edges['relations'].append(relation)
            return edges


    def get_relations_from_document(
            self, 
            document_id: str,
            graph_metadata: Optional[Dict] = None
    ) -> List[Relation]:
        if not graph_metadata:
            graph_metadata = {}
        if graph_metadata:
            graph_metadata.pop("doc_type", None)
        d_metadata_conditions = ' AND '.join([f"d.{key} = $metadata.{key}" for key in graph_metadata.keys()])
        n1_metadata_conditions = ' AND '.join([f"n1.{key} = $metadata.{key}" for key in graph_metadata.keys()])
        n2_metadata_conditions = ' AND '.join([f"n2.{key} = $metadata.{key}" for key in graph_metadata.keys()])

        # Helper function to format WHERE clause conditionally
        def format_where_clause(*conditions):
            non_empty_conditions = [cond for cond in conditions if cond]
            if non_empty_conditions:
                return "WHERE " + " AND ".join(non_empty_conditions)
            else:
                return ""

        query = f"""
        MATCH (d:Document {{id: $document_id}})
        {format_where_clause(d_metadata_conditions)}
        MATCH (n1)-[r {{document_id: $document_id}}]->(n2)
        {format_where_clause("NOT r:From", n1_metadata_conditions, n2_metadata_conditions)}
        RETURN n1, r, n2
        """

        with self.driver.session() as session:
            result = session.run(query, document_id=document_id, metadata=graph_metadata)
            relations = []
            for record in result:
                source_metadata = dict(record['n1'])
                edge_metadata = dict(record['r'])
                target_metadata = dict(record['n2'])
                node_1 = Node(
                    id=source_metadata.pop('id'),
                    name=source_metadata.pop('name'), 
                    metadata=source_metadata
                )
                edge = Edge(
                    id=edge_metadata.pop('id'),
                    name=edge_metadata.pop('name'), 
                    metadata=edge_metadata
                )
                node_2 = Node(
                    id=target_metadata.pop('id'),
                    name=target_metadata.pop('name'), 
                    metadata=target_metadata
                )
                relation = Relation(
                    node_1=node_1,
                    edge=edge,
                    node_2=node_2,
                )
                relations.append(relation)
            return relations


    def find_paths(
            self, 
            start_node: str, 
            end_node: str, 
            max_depth: int = 3,
            graph_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        if not graph_metadata:
            graph_metadata = {}
        if graph_metadata:
            graph_metadata.pop("doc_type", None)
        n_metadata_conditions = ' AND '.join([f"n.{key} = $metadata.{key}" for key in graph_metadata.keys()])

        # Helper function to format WHERE clause conditionally
        def format_where_clause(conditions):
            if conditions:
                return f"WHERE {conditions} AND NONE(n IN nodes(p) WHERE n:Document)"
            else:
                return "WHERE NONE(n IN nodes(p) WHERE n:Document)"

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH p=shortestPath((start {{name: $start_node}})-[*..{max_depth}]-(end {{name: $end_node}}))
                {format_where_clause(n_metadata_conditions)}
                RETURN nodes(p) AS nodes, relationships(p) AS relationships
                """, start_node=start_node, end_node=end_node, max_depth=max_depth, metadata=graph_metadata
            )
            output = {
                'relations': [],
                'records': []
            }
            for record in result:
                output['records'].append(record)
                nodes = record['nodes']
                relationships = record['relationships']
                for rel in relationships:
                    source_node = next(node for node in nodes if node.id == rel.start_node.id)
                    target_node = next(node for node in nodes if node.id == rel.end_node.id)
                    source_node = dict(source_node)
                    rel_type = rel.type
                    rel = dict(rel)
                    target_node = dict(target_node)
                    node_1 = Node(
                        id=source_node.pop('id'),
                        name=source_node.pop('name'),
                        metadata=dict(source_node)
                    )
                    edge = Edge(
                        name=rel['name'],
                        type=rel_type,
                        metadata=dict(rel)
                    )
                    node_2 = Node(
                        id=target_node.pop('id'),
                        name=target_node.pop('name'),
                        metadata=dict(target_node)
                    )
                    relation = Relation(
                        node_1=node_1,
                        edge=edge,
                        node_2=node_2,
                    )
                    output['relations'].append(relation)
            return output


    def _delete_from_graph(
            self, 
            metadata: Dict[str, Any]
    ) -> Dict[str, List[str]]:

        d_metadata_conditions = ' AND '.join([f"d.{key} = $metadata.{key}" for key in metadata.keys()])
        n_metadata_conditions = ' AND '.join([f"n.{key} = $metadata.{key}" for key in metadata.keys()])
        r_metadata_conditions = ' AND '.join([f"r.{key} = $metadata.{key}" for key in metadata.keys()])
        f_metadata_conditions = ' AND '.join([f"f.{key} = $metadata.{key}" for key in metadata.keys()])

        def format_where_clause(conditions):
            if conditions:
                return f"WHERE {conditions}"
            else:
                return ""

        with self.driver.session() as session:
            # Match and collect document IDs
            document_result = session.run(
                f"""
                MATCH (d:Document)
                {format_where_clause(d_metadata_conditions)}
                RETURN d.id as document_id
                """, metadata=metadata
            )
            document_ids = [record["document_id"] for record in document_result]

            # Match and collect node IDs
            node_result = session.run(
                f"""
                MATCH (n:Node)
                {format_where_clause(n_metadata_conditions)}
                RETURN n.id as node_id
                """, metadata=metadata
            )
            node_ids = [record["node_id"] for record in node_result]

            # Match and collect edge IDs
            edge_result = session.run(
                f"""
                MATCH ()-[r:Edge]->()
                {format_where_clause(r_metadata_conditions)}
                RETURN r.id as edge_id
                """, metadata=metadata
            )
            edge_ids = [record["edge_id"] for record in edge_result]

            # Delete documents, nodes, edges, and relationships
            delete_query = f"""
            MATCH (d:Document)
            {format_where_clause(d_metadata_conditions)}
            DETACH DELETE d
            WITH COUNT(d) AS deleted_docs

            MATCH (n:Node)
            {format_where_clause(n_metadata_conditions)}
            DETACH DELETE n
            WITH COUNT(n) AS deleted_nodes

            MATCH ()-[r:Edge]->()
            {format_where_clause(r_metadata_conditions)}
            DELETE r
            WITH COUNT(r) AS deleted_edges

            MATCH ()-[f:From]->()
            {format_where_clause(f_metadata_conditions)}
            DELETE f
            """

            session.run(delete_query, metadata=metadata)

            return {
                'document_ids': document_ids,
                'node_ids': node_ids,
                'edge_ids': edge_ids,
                'ids': document_ids + node_ids + edge_ids
            }


    def _delete_from_vectorstore(
            self,
            ids: List[str]
    ) -> Optional[bool]:
        output = self.vectorstore.delete(ids=list(set(ids)))
        return output


    def reset(
            self,
            metadata: Dict
    ) -> Optional[bool]:
        ids = self._delete_from_graph(metadata)['ids']

        if ids:
            output = self._delete_from_vectorstore(ids=ids)
            logger.debug(f"Deleted {len(ids)} vectors/rows")
        else:
            return True
        return output
