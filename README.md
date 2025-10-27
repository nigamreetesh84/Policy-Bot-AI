# 🤖 PolicyBot AI
### A Multi-Document RAG Assistant for Insurance Policies

## 🎯 Problem Statement
PolicyBot AI enables users to query multiple insurance policy documents and get accurate, cited answers instantly using Retrieval-Augmented Generation (RAG).

## 🧱 System Design
1. PDF Upload → Document Chunking  
2. Embedding Creation → ChromaDB Vector Store  
3. Semantic Retrieval → Reranking  
4. GPT-based Answer Generation → Cited Output  
5. User Feedback Logging

## 🧰 Tech Stack
Python · LangChain · ChromaDB · SentenceTransformers · OpenAI API · Streamlit

## ⚙️ How to Run
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key"
streamlit run app.py
