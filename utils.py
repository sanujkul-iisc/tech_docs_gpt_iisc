from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from pinecone import Pinecone
import streamlit as st
import openai
import os
from dotenv import load_dotenv
from langchain.schema import BaseRetriever, Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.load import dumps, loads
from pprint import pprint

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class ChatMessage:
    role: str
    content: str

class PineconeDBRetriever(BaseRetriever, BaseModel):
    """
    A custom LangChain retriever for Pinecone.
    """
    index_name: str
    pinecone_api_key: str
    namespace: str
    top_k: int = 5
    index: Any = Field(None, exclude=True)

    def __init__(self, **data):
        """
        Initializes the Pinecone client and index.
        """
        super().__init__(**data)
        pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = pc.Index(self.index_name)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        The core method to retrieve documents. LangChain's retriever system
        calls this method.

        Args:
            query (str): The user's question.

        Returns:
            List[Document]: A list of relevant documents from Pinecone.
        """
        # Pinecone's hosted embedding model will automatically embed the query text.
        results = self.index.search(
            namespace=self.namespace,
            query={
                "inputs": {"text": query},
                "top_k": self.top_k
            }
        )

        # Convert Pinecone's search results into LangChain Document objects.
        # TODO: Add additional fields as necessary.
        documents = []
        if results and 'result' in results and 'hits' in results['result']:
            for match in results['result']['hits']:
                # The actual text content is in the 'fields' dictionary
                page_content = match.get('fields', {}).get('text', '')
                # metadata = {"id": match.get("_id"), "score": match.get("_score")}
                metadata = {"id": match.get("_id")} # Removing score to allow easy serialization and help de-duplication

                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                documents.append(doc)

        return documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronous version of the document retrieval method.
        """
        # For simplicity, we'll just call the synchronous version.
        # For a production environment, you might want to use an async Pinecone client.
        return self._get_relevant_documents(query)
    
    

class RAGOrchestrator:
    """
    Orchestrates the RAG pipeline based on a given configuration.
    """
    def __init__(self, config: dict):
        """
        Initializes the orchestrator with a configuration dictionary.
        """
        self.config = config
        self.debug = config.get("debug", False)
        self.llm = ChatOpenAI(
            model=config.get("llm_model", "gpt-4"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        if config.get("retrieval_strategy") != "llm_only":
            self.retriever = PineconeDBRetriever(
                index_name=config.get("index_name"),
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                namespace=config.get("namespace"),
                top_k=config.get("top_k", 5)
            )

    def _print_debug(self, header: str, data: Any):
        if self.debug:
            print("\n" + "="*20)
            print(f"DEBUG: {header}")
            print("="*20)
            pprint(data)
        return data

    def _tap_and_log(self, x: dict) -> dict:
        self._print_debug("Final Context for LLM", x.get("context_str", "Context not available"))
        return x

    def _get_unique_union(self, documents: List[List[Document]]) -> List[Document]:
        """
        Takes a list of document lists, merges them, and removes duplicates.
        """
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    def _get_multi_query_chain(self):
        template = """You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | RunnableLambda(lambda x: [q for q in x if q.strip()])
            | RunnableLambda(lambda x: self._print_debug("Generated Queries", x))
        )

        retrieval_chain = generate_queries | self.retriever.map() | self._get_unique_union
        return retrieval_chain

    def _get_rag_fusion_chain(self):
        template = """You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | RunnableLambda(lambda x: [q for q in x if q.strip()])
            | RunnableLambda(lambda x: self._print_debug("Generated Queries", x))
        )

        def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Document]:
            fused_scores = {}
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    fused_scores[doc_str] += 1 / (rank + k)

            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            self._print_debug("Reranked Documents (RAG Fusion)", reranked_results)
            return [doc for doc, score in reranked_results]

        retrieval_chain = generate_queries | self.retriever.map() | reciprocal_rank_fusion
        return retrieval_chain

    def invoke(self, question: str) -> dict:
        """
        Builds and invokes the RAG chain based on the configuration.
        """
        strategy = self.config.get("retrieval_strategy", "simple")

        if strategy == "llm_only":
            self._print_debug("Strategy", "LLM Only (No RAG)")
            answer = self.llm.invoke(question).content
            return {
                "question": question,
                "answer": answer,
                "strategy": strategy,
                "context": "N/A"
            }

        if strategy == "multi_query":
            retrieval_chain = self._get_multi_query_chain()
        elif strategy == "rag_fusion":
            retrieval_chain = self._get_rag_fusion_chain()
        else:
            retrieval_chain = self.retriever

        final_prompt_template = """Answer the following question based only on the provided context.
        If the context doesn't contain enough information to answer the question, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        final_prompt = ChatPromptTemplate.from_template(final_prompt_template)

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        rag_steps = (
            RunnablePassthrough.assign(
                context_str=itemgetter("context") | RunnableLambda(format_docs)
            )
            | RunnableLambda(self._tap_and_log)
        )

        rag_chain = (
            {"context": retrieval_chain, "question": RunnablePassthrough()}
            | rag_steps
            | {
                "answer": (
                    lambda x: {"context": x["context_str"], "question": x["question"]}
                ) | final_prompt | self.llm | StrOutputParser(),
                "context": itemgetter("context"),
            }
        )

        result = rag_chain.invoke(question)

        return {
            "question": question,
            "answer": result['answer'],
            "strategy": strategy,
            "context": result['context']
        }

def initialize_pinecone(api_key: str) -> Optional[RAGOrchestrator]:
    """
    Initialize the RAG Orchestrator with Pinecone configuration.
    """
    if not api_key:
        st.warning("Please check your Pinecone API key.")
        return None
    
    try:
        config = {
            "debug": True,
            "llm_model": "gpt-4",
            "index_name": "test-index",
            "namespace": "example-namespace",
            "top_k": 5,
            "retrieval_strategy": "rag_fusion"  # Options: simple, multi_query, rag_fusion, llm_only
        }
        
        orchestrator = RAGOrchestrator(config)
        return orchestrator
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

def query_pinecone(orchestrator: RAGOrchestrator, query_text: str) -> str:
    """
    Query using the RAG Orchestrator and format the response.
    """
    try:
        result = orchestrator.invoke(query_text)
        return result["answer"]
    except Exception as e:
        st.error(f"Error: {str(e)}")
        raise Exception(f"Error querying Pinecone: {str(e)}")

def setup_index_with_data(orchestrator: RAGOrchestrator) -> None:
    """
    Set up the Pinecone index with initial data.
    """
    try:
        # Sample data
        data = [
            {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
            {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
            {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
            {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
            {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
            {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
        ]

        records = []
        for item in data:
            records.append({
                "id": item["id"],
                "fields": {"text": item["text"]}
            })

        orchestrator.retriever._index.upsert(
            vectors=records,
            namespace=orchestrator.retriever._namespace
        )
        st.success("Successfully uploaded data to Pinecone!")
        
    except Exception as e:
        st.error(f"Error upserting data: {str(e)}")
        raise Exception(f"Error upserting data: {str(e)}")
