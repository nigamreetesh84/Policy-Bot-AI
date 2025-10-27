# generate.py
from openai import OpenAI
from cache import SimpleCache
from search import retrieve, rerank, format_retrieved_docs
from config import OPENAI_MODEL
from open_api import api_key

client = OpenAI(api_key=api_key)

PROMPT_SYSTEM = """You are PolicyBot AI, an assistant that answers user questions 
strictly using provided insurance policy chunks.
Rules:
- Use only the provided chunks; do not hallucinate.
- If the answer is not in the context, say: "Answer not found in provided policies."
- Keep your answer concise, factual, and formatted for readability.
"""

def make_prompt(query, top_chunks):
    """Compact prompt to minimize token usage."""
    context = "\n".join([f"[{c['id']}] {c['text'][:600]}" for c in top_chunks])
    return f"Use only the context below to answer briefly and accurately.\n\n{context}\n\nQ: {query}\nA:"

def format_answer_with_citations(answer, top_chunks):
    """Append short reference section."""
    refs = []
    for c in top_chunks:
        src = c.get("meta", {}).get("source", "Policy Document")
        cid = c["id"]
        pg = c.get("meta", {}).get("page", "N/A")
        refs.append(f"- {src} [chunk: {cid}, page: {pg}]")
    return f"{answer}\n\n**References:**\n" + "\n".join(refs)

def answer_with_openai(query):
    """Retrieve, rerank, and generate compact grounded answer."""
    items = retrieve(query, top_k=20)
    top_r = rerank(query, items, top_n=5)

    prompt = make_prompt(query, top_r)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=400,
    )

    answer = response.choices[0].message.content.strip()
    formatted_context = format_retrieved_docs(top_r)
    final_answer = format_answer_with_citations(answer, top_r)

    return final_answer, formatted_context

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PolicyBot AI Answer Generator")
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    print(answer_with_openai(args.query))
