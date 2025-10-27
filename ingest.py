# ingest.py
import os, re
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", message="Cannot copy out of meta tensor")

from tqdm import tqdm
from chromadb import PersistentClient
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR, EMBED_MODEL

def clean_text(s: str) -> str:
    """Remove unwanted artifacts and normalize whitespace."""
    s = s.replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'This page left blank intentionally', '', s, flags=re.I)
    return s.strip()

def extract_pdf_chunks(pdf_path):
    """Load a PDF and split into cleaned text chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    texts, metadata = [], []
    for idx, chunk in enumerate(chunks):
        text = clean_text(chunk.page_content)
        meta = chunk.metadata.copy()
        meta["chunk_id"] = f"{os.path.basename(pdf_path)}_chunk_{idx}"
        meta["source"] = os.path.basename(pdf_path)
        texts.append(text)
        metadata.append(meta)
    return texts, metadata

def build_embeddings(texts):
    """Generate dense embeddings using SentenceTransformer."""
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def upsert_chroma(texts, metadata, embeddings):
    """Persist embeddings into ChromaDB."""
    client = PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("policy_chunks")
    ids = [m["chunk_id"] for m in metadata]
    collection.add(documents=texts, metadatas=metadata, embeddings=embeddings.tolist(), ids=ids)
    print(f"✅ Stored {len(texts)} chunks to {CHROMA_DIR}")

def process_multiple_pdfs(pdf_paths):
    """End-to-end ingestion for multiple PDFs."""
    all_texts, all_meta = [], []
    for pdf in pdf_paths:
        texts, meta = extract_pdf_chunks(pdf)
        all_texts.extend(texts)
        all_meta.extend(meta)
    embeddings = build_embeddings(all_texts)
    upsert_chroma(all_texts, all_meta, embeddings)
