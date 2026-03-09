# RAG-Chatbot-on-Human-Chakras
An assistant that answers queries on how human chakras and energy field works with citations
This project implements a Retrieval-Augmented Generation (RAG) chatbot. The system retrieves relevant information from a curated document corpus and generates grounded responses using an LLM. The goal is to reduce hallucination and ensure answers are based only on the provided knowledge source

# Live demo: https://rag-chatbot-on-human-chakras-835cnokags38vzqxyg8hgs.streamlit.app/

# Substack link: https://substack.com/@vanibatra/p-190170072

# Architecture:
  User question > Embedding generation > FAISS vector search using semantic search > Relevant context retrieval > LLM to generate answer via GPT-4o-min > Grounded answer+source citation
  
  # Tech Stack:
    Python language for code language
    Lanchain for building RAG pipeline
    Open AI Embeddings to create embeddings of the chunks from the source documents
    FAISS vector database to store and search embeddings
    Streamlit UI and Streamlit Cloud for deployment
    
# Key features:
  Retrieval Augment Generation with 5 PDFs as input documents
  Vector search using FAISS via a semantic approach, chunking of 1000 words with 200-word overlap
  Source citation for transparency
  Conversational memory
  Low hallucination for grounded prompting
  Public deployment via Streamlit Cloud
  
# Repository Structure
app.py > streamlit application
faiss_index/ > vector database
index.faiss > FAISS index
index.pkl > metadata
requirements.txt > dependencies
readme.md > project documentation

# How is hallucination reduced?
Retrieving relevant documents via FAISS vector search, a semantic search approach
Passing retrieved context to LLM
Constraining the model with a prompt
Setting temperature = 0 for deterministic responses

# Limitations
Retrieval uses only a semantic search approach, instead of a hybrid approach
Chunking size set to 1000 words with 200 words overlap instead of any other convenient chunking approach
No reranking of retrieved chunks
User query rewriting not implemented

# Future improvements (Advanced RAG)
Query rewriting for better retrieval
Hybrid search
Different chunking approach
Chunk reranking
Evaluation metrics of the RAG product
