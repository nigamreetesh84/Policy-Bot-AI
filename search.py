# search.py
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from cache import SimpleCache
from config import CHROMA_DIR, EMBED_MODEL, CROSS_ENCODER, CACHE_PATH

# Initialize persistent components
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("policy_chunks")
embed_model = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(CROSS_ENCODER)
cache = SimpleCache(filename=CACHE_PATH)

def retrieve(query, top_k=20):
    """Retrieve top-k semantically similar chunks from ChromaDB."""
    cached = cache.get(query)
    if cached:
        return cached

    q_emb = embed_model.encode(query)
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
    items = [
        {"id": id_, "text": doc, "meta": meta}
        for doc, meta, id_ in zip(results['documents'][0], results['metadatas'][0], results['ids'][0])
    ]
    cache.set(query, items)
    return items

def rerank(query, items, top_n=5):
    """Re-rank retrieved chunks using a cross-encoder model."""
    pairs = [(query, it["text"]) for it in items]
    scores = reranker.predict(pairs)
    scored = [{**it, "score": float(s)} for it, s in zip(items, scores)]
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

if __name__ == "__main__":
    q = "What is the scheduled benefit for all members?"
    items = retrieve(q)
    top3 = rerank(q, items)
    for i, r in enumerate(top3, 1):
        print(f"{i}. {r['id']}  (score={r['score']:.3f})\n{r['text'][:200]}...\n")
