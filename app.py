import streamlit as st
from ingest import process_multiple_pdfs
from generate import answer_with_openai
import os

# App title and description
st.set_page_config(page_title="PolicyBot AI", page_icon="🤖")
st.title("🤖 PolicyBot AI")
st.caption("An AI-powered RAG assistant for understanding insurance policy documents.")

# Directory setup
os.makedirs("data", exist_ok=True)
os.makedirs("feedback", exist_ok=True)

# Step 1: Upload multiple PDFs
st.header("📂 Upload Policy Documents")
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    paths = []
    for file in uploaded_files:
        save_path = os.path.join("data", file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(save_path)
    process_multiple_pdfs(paths)
    st.success(f"✅ {len(paths)} document(s) ingested successfully!")

# Step 2: Ask a question
st.header("💬 Ask a Question")
query = st.text_input("Type your question here (e.g., 'What is the benefit limit for accidental death?')")

if st.button("Generate Answer") and query:
    with st.spinner("Generating answer using PolicyBot AI..."):
        answer = answer_with_openai(query)
        st.markdown(answer)

    # Step 3: Feedback section
    st.markdown("---")
    st.subheader("🗳️ Was this answer helpful?")
    col1, col2 = st.columns(2)
    if col1.button("👍 Helpful"):
        with open("feedback/feedback_log.csv", "a") as f:
            f.write(f"{query},Helpful\n")
        st.toast("Thanks for your feedback!", icon="✅")

    if col2.button("👎 Not Helpful"):
        with open("feedback/feedback_log.csv", "a") as f:
            f.write(f"{query},Not Helpful\n")
        st.toast("Thanks! We'll keep improving.", icon="⚙️")
