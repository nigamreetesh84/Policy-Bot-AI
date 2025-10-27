
# config.py
"""
Central configuration for PolicyBot AI project.
"""

# ---------------------- Chunking Parameters ----------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------- Embedding & Vector DB ----------------------
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHROMA_DIR = "chroma_db"

# ---------------------- Cross-Encoder for Reranking ----------------------
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------- Cache ----------------------
CACHE_PATH = "cache/db"

# ---------------------- OpenAI Model ----------------------
# You can switch to another model here (like "gpt-4o" or "gpt-4o-mini")
OPENAI_MODEL = "gpt-4o-mini"
