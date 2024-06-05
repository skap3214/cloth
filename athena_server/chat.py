from langchain_groq import ChatGroq
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from typing import Generator, Any, Dict, List, Optional, Literal
from .types import ChatResponse
from cloth.neo4j_graph import Neo4jGraphstore
from cloth import Relation

LLM = ChatGroq(model="llama3-70b-8192", temperature=0.2)

SYSTEM_PROMPT = """\
You are a personal assistant that is tasked with helping the user answer questions based on a knowledge graph.

Here is a summary of the relevant information from the knowledge graph:
{information}\

Based on the information given to you, answer the user's query. Answer in markdown.
"""

MAP_PROMPT = """You are given Source, Link, Target pairs from a part of a knowledge graph within the delimiters:
```
{docs}
```
Based on the pairs from the knowledge graph, provide a concise summary of all the source, links and targets. \
Include all the Source, Link, Target key-words and sentences in your concise summary. Respond with ONLY the summary and nothing else \
Start your response with the summary:"""

REDUCE_PROMPT = """You are given a set of summaries that were extracted from a knowledge graph within the delimiters:
```
{docs}
```
Based on the summaries, provide a concise summary. \
Respond with ONLY the summary and nothing else. Start your response with the summary:"""

CHAT_MEMORY = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)

class Agent:
    def __init__(
        self,
        llm: BaseChatModel = LLM,
        system_prompt: str = SYSTEM_PROMPT,
        map_prompt: str = MAP_PROMPT,
        reduce_prompt: str = REDUCE_PROMPT,
        chat_memory = CHAT_MEMORY,
        graph: Optional[Neo4jGraphstore] = None
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.chat_memory = chat_memory
        self.memory_key = chat_memory.memory_key
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name=self.memory_key, optional=True),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.agent = self.prompt | self.llm | StrOutputParser()

        # TODO: Langchain map reduce summarizer SUCKS
        map_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(map_prompt)
        )
        reduce_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(reduce_prompt)
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name='docs',
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=2500
        )
        self.summary_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name='docs',
            return_intermediate_steps=False
        )
        self.graph = graph or Neo4jGraphstore()


    def _chat(self, query: str, information: str, save_chat_history: bool = True) -> Generator[str, Any, None]:

        llm_input = {
            'information': information,
            'input': query,
            'chat_history': self.chat_memory.load_memory_variables({})[self.memory_key]
        }
        stream = self.agent.stream(llm_input)
        response = ""
        for chunk in stream:
            response += chunk
            yield chunk

        if save_chat_history:
            self.chat_memory.save_context({'input': query}, {'output': response})
    
    def _relation_to_string(self, relation: Relation):
        template = f"""Source: {relation.node_1.name}Link: {relation.edge.name}Target: {relation.node_2.name}"""
        return template

    def _node_search(self, query: str, metadata: Dict, k: int = 3) -> Dict[str, Any]:
        # Search based on most similar nodes to the query
        similar_nodes = self.graph.similarity_search(query, doc_type='node', k=k, filter=metadata)
        all_documents: List[Document] = []
        all_relations: List[Relation] = []
        for node in similar_nodes:
            adj = self.graph.get_adjacent_edges(node, graph_metadata=metadata)
            outgoing = adj['outgoing']
            incoming = adj['incoming']
            both = outgoing + incoming
            all_relations.extend(both)
            all_documents.extend(Document(page_content=self._relation_to_string(rel)) for rel in both)
        
        return {
            'nodes': similar_nodes,
            'documents': all_documents,
            'relations': all_relations
        }
    
    def chat_node(self, query: str, metadata: Dict):
        # Perform search over nodes and summarize any documents
        output = self._node_search(query, metadata=metadata)
        nodes = output['nodes']
        documents = output['documents']
        relations = output['relations']
        bulleted_nodes = '\n\n'.join(['- Node: ' + node for node in nodes])
        yield ChatResponse(
            sources=relations,
            type='intermediate',
            delta=f"{bulleted_nodes}"
        )
        # Summarize
        summary = self.summary_chain.invoke(documents)
        yield ChatResponse(
            sources=relations,
            type='intermediate',
            delta=f" *{summary['output_text']}* "
        )
        output = ""
        for chunk in self._chat(query, summary, False):
            output += chunk
        yield ChatResponse(
            sources=relations,
            type='output',
            delta=f"{output}"
        )

    def _edge_search(self, query: str, metadata: Dict, k: int = 7) -> Dict[str, Any]:
        # Search based on most similar nodes to the query
        similar_edges = self.graph.similarity_search(query, doc_type='edge', k=k, filter=metadata)
        all_documents: List[Document] = []
        all_relations: List[Relation] = []
        for edge in similar_edges:
            adj = self.graph.get_adjacent_nodes(edge, graph_metadata=metadata)
            relations = adj['relations']
            all_relations.extend(relations)
            all_documents.extend(Document(page_content=self._relation_to_string(rel)) for rel in relations)
        
        return {
            'edges': similar_edges,
            'documents': all_documents,
            'relations': all_relations
        }
    
    def chat_edge(self, query: str, metadata: Dict):
        # Perform search over edges and summarize any documents
        output = self._edge_search(query, metadata=metadata)
        edges = output['edges']
        documents = output['documents']
        relations = output['relations']
        yield ChatResponse(
            sources=relations,
            type='intermediate',
            delta=f"Summarizing the surroundings of these edges: {', '.join(edges)}"
        )
        # Summarize
        summary = self.summary_chain.invoke(documents)
        yield ChatResponse(
            sources=relations,
            type='intermediate',
            delta=f"Summary of edges:{summary['output_text']}"
        )
        output = ""
        for chunk in self._chat(query, summary, False):
            output += chunk
        yield ChatResponse(
            sources=relations,
            type='output',
            delta=output
        )

    def _raw_search(self, query: str, metadata: Dict, k: int = 3):
        docs = self.graph.similarity_search(query, doc_type='raw', k=k, filter=metadata)
        all_documents: List[Document] = []
        all_relations: List[Relation] = []
        for doc in docs:
            adj = self.graph.get_adjacent_nodes(doc, graph_metadata=metadata)
            relations = adj['relations']
            all_relations.extend(relations)
            all_documents.extend(Document(page_content=self._relation_to_string(rel)) for rel in relations)
        
        return {
            'raws': docs,
            'documents': all_documents,
            'relations': all_relations
        }
    
    def chat_raw(self, query: str, metadata: Dict):
        # Perform search over edges and summarize any documents
        output = self._raw_search(query, metadata=metadata)
        raws = output['raws']
        documents = output['documents']
        relations = output['relations']
        result = "Similarity Search:"
        for i, r in enumerate(raws, 1):
            result += f"{i}. {r[:10]}..."

        result = result.rstrip()  # Remove the trailing newline if necessary

        yield ChatResponse(
            sources=relations,
            type='intermediate',
            delta=result + ''
        )
        # Summarize
        summary = self.summary_chain.invoke(documents)
        yield ChatResponse(
            sources=relations,
            type='intermediate',
            delta=f"Summary of raw docs:{summary['output_text']}"
        )

        output = ""
        for chunk in self._chat(query, summary, False):
            output += chunk
        yield ChatResponse(
            sources=relations,
            type='output',
            delta=output
        )

    def chat(self, query: str, type: Literal['node', 'edge', 'raw'], metadata: Dict):
        if type == 'node':
            return self.chat_node(query, metadata=metadata)
        elif type == 'edge':
            return self.chat_edge(query, metadata=metadata)
        else:
            return self.chat_raw(query, metadata=metadata)
# agent = Agent()

# while True:
#     query = input("Chat: ")
#     print("Response:")
#     for token in agent.chat(query):
#         print(token, end='', flush=True)
#     print("")
