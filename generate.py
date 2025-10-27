# generate.py
import os
from openai import OpenAI
from search import retrieve, rerank
from open_api import api_key
client = OpenAI(api_key=api_key)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("🤖 Initialized OpenAI client.")
PROMPT_SYSTEM = """You are PolicyBot AI, an assistant that answers user questions 
strictly using provided insurance policy chunks. 
Rules:
- Use only the provided chunks; do not hallucinate.
- If the answer is not in the context, say: "Answer not found in provided policies."
- Always include a short summary and citations.
"""

def make_prompt(query, top_chunks):
    """Combine retrieved chunks and user query into a single prompt."""
    context = "\n\n---\n".join([f"[{c['id']}] {c['text']}" for c in top_chunks])
    return f"{PROMPT_SYSTEM}\n\nContext:\n{context}\n\nQ: {query}\nA:"

def format_answer_with_citations(answer, top_chunks):
    """Append citation references from retrieved chunk metadata."""
    refs = []
    for c in top_chunks:
        src = c["meta"].get("source", "Policy Document")
        cid = c["id"]
        pg = c["meta"].get("page", "N/A")
        refs.append(f"- {src} [chunk_id: {cid}, page: {pg}]")
    return f"{answer}\n\n**References:**\n" + "\n".join(refs)

def answer_with_openai(query):
    """Retrieve, rerank, and generate a grounded answer with GPT."""
    items = retrieve(query, top_k=20)
    top_r = rerank(query, items, top_n=5)
    prompt = make_prompt(query, top_r)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()
    return format_answer_with_citations(answer, top_r)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PolicyBot AI Answer Generator")
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    print(answer_with_openai(args.query))
