

import os, tempfile, uuid
from tqdm import tqdm
from chromadb import PersistentClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def extract_pdf_chunks(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    return texts, metas

def process_multiple_pdfs(pdf_files):
    """Process and embed PDFs into Chroma DB."""
    client = PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("policy_chunks")
    model = SentenceTransformer(EMBED_MODEL)

    all_texts, all_meta, all_ids = [], [], []

    for f in tqdm(pdf_files, desc="Batches"):
        # Handle UploadedFile or file path
        if hasattr(f, "getbuffer"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                fpath = tmp.name
        else:
            fpath = f

        texts, meta = extract_pdf_chunks(fpath)

        # Generate unique IDs for chunks
        chunk_ids = [str(uuid.uuid4()) for _ in texts]

        all_texts.extend(texts)
        all_meta.extend(meta)
        all_ids.extend(chunk_ids)

        try:
            os.remove(fpath)
        except:
            pass

    embeds = model.encode(all_texts, show_progress_bar=True)

    # ✅ FIX: include 'ids' argument
    collection.add(
        ids=all_ids,
        documents=all_texts,
        embeddings=embeds.tolist(),
        metadatas=all_meta,
    )

    print(f"✅ Stored {len(all_texts)} chunks to {CHROMA_DIR}")
