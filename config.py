# config.py
"""
Central configuration for PolicyBot AI project.
"""

# Chunking parameters (can tune for document size)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding and DB paths
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHROMA_DIR = "chroma_db"

# Cross encoder for reranking
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cache file path
CACHE_PATH = "cache/db"
