import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, EMBED_MODEL
from sentence_transformers import CrossEncoder
from config import CROSS_ENCODER

def rerank(query, retrieved_docs, top_n=5):
    """
    Re-rank retrieved chunks using a cross-encoder for higher semantic precision.
    Returns top_n highest-scoring chunks.
    """
    if not retrieved_docs:
        return []

    # Load cross-encoder once (cached per process)
    model = CrossEncoder(CROSS_ENCODER)

    pairs = [(query, d["text"]) for d in retrieved_docs]
    scores = model.predict(pairs)

    for i, s in enumerate(scores):
        retrieved_docs[i]["rerank_score"] = float(s)

    # Sort by descending score (higher = more relevant)
    sorted_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_docs[:top_n]


def retrieve(query, top_k=5):
    """Return top-k most relevant text chunks for a query."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("policy_chunks")
    model = SentenceTransformer(EMBED_MODEL)

    query_emb = model.encode([query])
    results = collection.query(query_embeddings=query_emb, n_results=top_k)

    docs = results["documents"][0]
    ids = results["ids"][0]
    scores = results["distances"][0]  # smaller = better

    structured = [{"id": i, "text": d, "score": s} for i, d, s in zip(ids, docs, scores)]
    return structured

def format_retrieved_docs(docs):
    """
    Takes list of retrieved document chunks and returns pretty formatted markdown.
    """
    formatted_list = []
    for d in docs:
        meta = d.get("meta", {})
        text = d.get("text", "").strip().replace("\n", " ")

        # Extract useful metadata safely
        source = meta.get("source", "Unknown Source")
        page = meta.get("page", "N/A")
        title = meta.get("title", "")
        company = meta.get("company", "")
        creation = meta.get("creationdate", "")[:10]  # just date part

        formatted = f"""
            **📄 {source} (Page {page})**

            > _{text[:500]}..._

            **Metadata:**
            - **Title:** {title or "N/A"}
            - **Company:** {company or "N/A"}
            - **Created:** {creation or "N/A"}
"""
        formatted_list.append(formatted)
    return "\n\n---\n\n".join(formatted_list)
