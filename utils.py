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
from pydantic import BaseModel, Field
from sentence_transformers.cross_encoder import CrossEncoder
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

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
    def __init__(self, config: dict, OPENAI_API_KEY, PINECONE_API_KEY):
        """
        Initializes the orchestrator with a configuration dictionary.

        Args:
            config (dict): A dictionary containing settings for the RAG pipeline,
                           such as model name, index name, and retrieval strategy.
        """
        self.config = config
        self.debug = config.get("debug", False)
        self.llm = ChatOpenAI(
            model=config.get("llm_model", "gpt-4o-mini"),
            api_key=OPENAI_API_KEY
        )
        if config.get("retrieval_strategy") != "llm_only":
            self.retriever = PineconeDBRetriever(
                index_name=config.get("index_name"),
                pinecone_api_key=PINECONE_API_KEY,
                namespace=config.get("namespace"),
                top_k=config.get("top_k", 5)
            )

    # --- Debugging Helper ---
    def _print_debug(self, header: str, data: Any):
        if self.debug:
            print("\n" + "="*20)
            print(f"DEBUG: {header}")
            print("="*20)
            pprint(data)
        return data # Pass data through unchanged

    def _tap_and_log(self, x: dict) -> dict:
        """
        A helper method to print debug info and pass the input dictionary through unchanged.
        """
        self._print_debug("Final Context for LLM", x.get("context_str", "Context not available"))
        return x

    def _get_unique_union(self, documents: list[list]) -> List[Document]:
        """
        Takes a list of document lists, merges them, and removes duplicates.
        """
        # Serialize each document to a string to make them hashable
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Use a set to get unique serialized documents
        unique_docs = list(set(flattened_docs))
        # Deserialize unique documents back into Document objects
        return [loads(doc) for doc in unique_docs]

    # --- Retrieval Strategy Helpers ---

    def _get_multi_query_chain(self):
        # Builds a chain that generates multiple queries and retrieves documents for each.
        # 1. Prompt for generating multiple queries
        template = """You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the cosine-based similarity search.
        Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        # 3. The chain for generating and retrieving
        generate_queries = (
            prompt_perspectives
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | RunnableLambda(lambda x: [q for q in x if q.strip()])
            | RunnableLambda(lambda x: self._print_debug("Generated Queries", x))
        )

        retrieval_chain = generate_queries | self.retriever.map() | self._get_unique_union | RunnableLambda(lambda docs: self._print_debug("Retrieved Documents", docs))
        return retrieval_chain

    def _get_rag_fusion_chain(self):
        """Builds a chain for RAG Fusion with reciprocal rank fusion."""
        # 1. The multi-query generation is the same as above
        template = """You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the cosine-based similarity search.
        Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | RunnableLambda(lambda x: [q for q in x if q.strip()])
            | RunnableLambda(lambda x: self._print_debug("Generated Queries", x))
        )

        # 2. Reranking with Reciprocal Rank Fusion
        def reciprocal_rank_fusion(results: list[list], k=60):
            fused_scores = {}
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    fused_scores[doc_str] += 1 / (rank + k)

            # .item() converts [doc_str: score] pairs to a list of tuples [doc_str, score]
            # Sort by score in descending order (reverse=True)
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            self._print_debug("Reranked Documents (RAG Fusion)", reranked_results)
            # Return only the documents, not the scores
            return [doc for doc, score in reranked_results]

        # 3. The RAG Fusion chain
        retrieval_chain = generate_queries | self.retriever.map() | reciprocal_rank_fusion
        return retrieval_chain

    def _get_decomposition_chain(self):
        # Builds a chain that decomposes a question into sub-questions.
        # 1. Prompt for generating sub-questions
        decomposition_template = """You are a helpful assistant that generates multiple sub-questions
        related to an input question. The goal is to break down the input into a set of sub-problems
        that can be answered in isolation. Generate multiple search queries related to: {question}
        Output (separated by newlines):"""
        prompt_decomposition = ChatPromptTemplate.from_template(decomposition_template)

        # 2. Chain to generate and clean up sub-questions
        generate_queries_decomposition = (
            prompt_decomposition
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | RunnableLambda(lambda x: [q for q in x if q.strip()])
            | RunnableLambda(lambda x: self._print_debug("Decomposed Sub-questions", x))
        )

        # 3. The full retrieval chain using the decomposed questions
        retrieval_chain = (
            generate_queries_decomposition
            | self.retriever.map()
            | self._get_unique_union
            | RunnableLambda(lambda docs: self._print_debug("Retrieved Documents (Decomposition)", docs)
            ))
        return retrieval_chain

    def _get_step_back_chain(self):
        # Builds a chain that generates a general, "stepped-back" question and retrieves documents for it.
        # 1. Prompt to generate a more general, "stepped-back" question
        step_back_template = """You are an expert at world knowledge. Your task is to step back and
        paraphrase a question to a more generic step-back question, which is easier to answer.

        Here are a few examples:
        Original Question: What is the C29x CPU architecture in the F29H85x microcontroller?
        Step-Back Question: What are the technical specifications of the C29x CPU architecture?

        Original Question: Which TI device was recommended for automotive radar in the 2023 safety seminar?
        Step-Back Question: What are some common TI devices used for automotive radar applications?

        Original Question: {question}
        Step-Back Question:"""
        prompt_step_back = ChatPromptTemplate.from_template(step_back_template)

        # 2. Chain to generate the new question
        generate_step_back_query = (
            prompt_step_back
            | self.llm
            | StrOutputParser()
            #| (lambda x: x.split("\n"))
            #| RunnableLambda(lambda x: [q for q in x if q.strip()])
            | RunnableLambda(lambda x: self._print_debug("Generated Step-Back Question", x))
        )

        # 3. The full retrieval chain using the new question
        # This takes the original question, generates a new one, and retrieves docs with it
        retrieval_chain = generate_step_back_query | self.retriever | RunnableLambda(lambda docs: self._print_debug("Retrieved Documents (Step back)", docs))
        return retrieval_chain

    def _get_hyde_chain(self):
        # Builds a chain that generates a hypothetical document and uses it for retrieval.

        # 1. Prompt to generate a hypothetical document (a plausible answer)
        hyde_template = """Please write a passage to answer the user's question.
        This passage should be detailed and informative, as if it came from a technical document.
        The purpose is to create a rich text for a vector search.

        Question: {question}
        Passage:"""
        prompt_hyde = ChatPromptTemplate.from_template(hyde_template)

        # 2. Chain to generate the hypothetical document
        generate_hyde_document = (
            prompt_hyde
            | self.llm
            | StrOutputParser()
            | RunnableLambda(lambda x: self._print_debug("Generated Hypothetical Document", x))
        )

        # 3. The full retrieval chain: generate a hypothetical doc, then retrieve with it
        retrieval_chain = generate_hyde_document | self.retriever | RunnableLambda(lambda docs: self._print_debug("Retrieved Documents (HyDE)", docs))
        return retrieval_chain

    # --- Post Retrieval Processing Helpers ---
    
    def _get_st_reranking_chain(self):
        """
        Creates a Runnable that performs semantic re-ranking using a
        Cross-Encoder model from the sentence-transformers library.
        """
        # Initialize a cross-encoder model. This model is lightweight and effective.
        # It will be downloaded from the Hugging Face Hub on first use.
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        def rerank_docs(inputs: dict):
            documents = inputs.get("documents", [])
            query = inputs.get("query", "")
            if not documents or not query:
                return []

            self._print_debug(f"Documents going INTO Re-ranker ({len(documents)} docs)", documents)

            # 1. Create pairs of [query, passage] for the cross-encoder
            sentence_pairs = [(query, doc.page_content) for doc in documents]

            # 2. Predict the relevance scores. The output is a list of scores.
            scores = model.predict(sentence_pairs)

            # 3. Combine the original documents with their new scores
            scored_docs = list(zip(scores, documents))

            # 4. Sort the documents by score in descending order
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # 5. Extract the documents and limit by top_n
            reranked_docs = [doc for score, doc in scored_docs]
            configured_top_n = self.config.get("reranker_top_n", 5)
            effective_top_n = min(configured_top_n, len(reranked_docs))
            final_docs_to_return = reranked_docs[:effective_top_n]
            
            self._print_debug(f"Documents COMING OUT of Re-ranker ({len(final_docs_to_return)} docs)", final_docs_to_return)
            return final_docs_to_return

        return RunnableLambda(rerank_docs)
    
    def _get_contextual_compression_retriever(self, base_retriever):
        """
        Takes a base retriever and wraps it with a compressor.
        """
        # 1. Initialize the compressor. This component uses an LLM to read each retrieved document and extract only the sentences relevant to the query.
        compressor = LLMChainExtractor.from_llm(self.llm)

        # 2. Create the compression retriever. This is a wrapper that first runs the base_retriever, then passes the results to the compressor.
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        self._print_debug("Contextual Compression Retriever", "Initialized and ready.")
        return compression_retriever    

    # --- Propmt Stregy Helpers ---
    def _get_final_prompt(self):
        """
        Selects and returns the final prompt template based on the config.
        """
        prompt_strategy = self.config.get("prompt_strategy", "strict_context")
        
        if prompt_strategy == "permissive_context":
            # This prompt allows the LLM to use its own knowledge
            template = """You are a helpful expert assistant for Texas Instruments products.
            Answer the user's question based on the context provided.
            If the context is not sufficient to answer the question, use your own knowledge to provide a comprehensive answer,
            but you must state that the information comes from your general knowledge.
            
            Context: {context}
            Question: {question}
            """
        else: # Default to "strict_context"
            # This prompt forces the LLM to only use the provided documents
            template = """Answer the following question based ONLY on the provided context.
            If the answer is not available in the context, you must say "Based on the provided context, I cannot answer this question."
            
            Context: {context}
            Question: {question}
            """
            
        self._print_debug(f"Using Prompt Strategy: {prompt_strategy}", template)
        return ChatPromptTemplate.from_template(template)
    
    def invoke(self, question: str) -> dict:
        """
        Builds and invokes the RAG chain based on the configuration.

        Args:
            question (str): The user's question.

        Returns:
            dict: A dictionary containing the question, retrieved context, and the final answer.
        """
        strategy = self.config.get("retrieval_strategy", "simple")
        post_processing_strategy = self.config.get("post_retrieval_processing", "none")
        prompt_strategy = self.config.get("prompt_strategy", "strict_context")

        # LLM_only strategy does not use retrieval
        if strategy == "llm_only":
            self._print_debug("Strategy", "LLM Only (No RAG)")
            answer = self.llm.invoke(question).content
            return {
                "question": question,
                "answer": answer,
                "strategy": strategy,
                "context": "N/A" # No context was used
            }

        # --- For RAG-based strategies ---

        # Select the base retrieval chain (gets the initial list of documents)
        if strategy == "multi_query":
            base_retrieval_chain = self._get_multi_query_chain()
        elif strategy == "rag_fusion":
            base_retrieval_chain = self._get_rag_fusion_chain()
        elif strategy == "decomposition":
            base_retrieval_chain = self._get_decomposition_chain()
        elif strategy == "step_back":
            base_retrieval_chain = self._get_step_back_chain()
        elif strategy == "hyde":
            base_retrieval_chain = self._get_hyde_chain()
        # -----------------------------
        else: # Default to simple retrieval
            base_retrieval_chain = self.retriever | RunnableLambda(
                lambda docs: self._print_debug("Retrieved Documents (Simple)", docs)
            )

        # Conditionally apply post-processing
        # If no post-processing is specified, use the base retrieval chain as is
        final_retrieval_chain = base_retrieval_chain
        
        if "semantic_re_ranking" in post_processing_strategy:
            # The re-ranker needs both docs and query
            reranker_chain = {"documents": final_retrieval_chain, "query": RunnablePassthrough()} | self._get_st_reranking_chain()
            final_retrieval_chain = reranker_chain
        
        # Conditionally apply the compression layer
        if "contextual_compression" in post_processing_strategy:
            # The compression retriever wraps the base retriever
            final_retrieval_chain = self._get_contextual_compression_retriever(final_retrieval_chain)


        # Final chains for invoking the LLM
        final_prompt = self._get_final_prompt()

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        context_formatter = (
            RunnablePassthrough.assign(
                context_str=itemgetter("context") | RunnableLambda(format_docs)
            )
            #| RunnableLambda(self._tap_and_log)
        )        
        
        rag_chain = (
            {"context": final_retrieval_chain, "question": RunnablePassthrough()}
            | context_formatter
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
            "strategy": f"{strategy} RAG + {post_processing_strategy} Post-Retrieval Processing + {prompt_strategy} Prompt Strategy",
            "context": result['context']
        }

def initialize_agent(OPENAI_API_KEY, PINECONE_API_KEY, retrieval_strategy="simple", post_retrieval_processing='none', prompt_strategy='strict_context') -> Optional[RAGOrchestrator]:
    """
    Initialize the RAG Orchestrator with Pinecone configuration.
    """
    if not PINECONE_API_KEY:
        st.warning("Please check your Pinecone API key.")
        return None
    
    try:
        config = {
            "llm_model": "gpt-4o-mini",
    	    "retrieval_strategy": retrieval_strategy,
    	    "post_retrieval_processing": post_retrieval_processing,
    	    "prompt_strategy": prompt_strategy,
    	    "index_name": "swru526-index",
    	    "namespace": "example-namespace",
    	    "top_k": 10,
    	    "reranker_top_n": 5,  # This is for the re-ranker, if used
    	    "debug": True  # <-- Enable debugging
        }
        #config = {
        #    "debug": True,
        #    "llm_model": "gpt-4",
        #    "index_name": "test-index",
        #    "namespace": "example-namespace",
        #    "top_k": 5,
        #    "retrieval_strategy": "rag_fusion"  # Options: simple, multi_query, rag_fusion, llm_only
        #}
 
        orchestrator = RAGOrchestrator(config, OPENAI_API_KEY, PINECONE_API_KEY)
        return orchestrator
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

def query_pinecone(orchestrator: RAGOrchestrator, query_text: str, retrieval_strategy, post_retrieval_processing,prompt_strategy) -> str:
    """
    Query using the RAG Orchestrator and format the response.
    """
    try:
        result = orchestrator.invoke(query_text)
        return result["answer"]
    except Exception as e:
        st.error(f"Error: {str(e)}")
        raise Exception(f"Error querying Pinecone: {str(e)}")

# def setup_index_with_data(orchestrator: RAGOrchestrator) -> None:
#     """
#     Set up the Pinecone index with initial data.
#     """
#     try:
#         # Sample data
#         data = [
#             {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
#             {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
#             {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
#             {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
#             {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
#             {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
#         ]

#         records = []
#         for item in data:
#             records.append({
#                 "id": item["id"],
#                 "fields": {"text": item["text"]}
#             })

#         orchestrator.retriever._index.upsert(
#             vectors=records,
#             namespace=orchestrator.retriever._namespace
#         )
#         st.success("Successfully uploaded data to Pinecone!")
        
#     except Exception as e:
#         st.error(f"Error upserting data: {str(e)}")
#         raise Exception(f"Error upserting data: {str(e)}")
