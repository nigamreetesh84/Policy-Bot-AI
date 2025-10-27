import streamlit as st
import time, tempfile, os, numpy as np, pandas as pd
from ingest import process_multiple_pdfs
from generate import answer_with_openai
from cache import SimpleCache
from search import retrieve

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="PolicyBot AI", page_icon="🤖", layout="wide")
st.title("🤖 PolicyBot AI: Multi-Document RAG Assistant")
st.caption("Upload insurance policies and ask questions. The assistant retrieves relevant sections and answers intelligently.")

st.markdown("""
<style>
.stExpander {border:1px solid #ccc;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("📂 Upload Policy Documents")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files and st.sidebar.button("🚀 Process Documents"):
    temp_paths = []
    with st.spinner("Processing and embedding PDFs..."):
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getbuffer())
                temp_paths.append(tmp.name)
        process_multiple_pdfs(temp_paths)
        for p in temp_paths:
            try: os.remove(p)
            except: pass
    st.session_state.processed = True
    st.sidebar.success("✅ Documents processed successfully!")

# ---------------------- MAIN CHAT SECTION ----------------------
st.subheader("💬 Ask a Question")

if not st.session_state.processed:
    st.info("Please upload and process policy documents first.")
else:
    user_input = st.text_input("Ask your question about the uploaded policies:")
    if user_input:
        cache = SimpleCache()
        start = time.time()

        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        cached = cache.get(user_input)
        if cached:
            # Support both old (string) and new (tuple) cache formats
            if isinstance(cached, tuple) and len(cached) == 2:
                answer, formatted_context = cached
            else:
                answer, formatted_context = cached, ""
            cache_status = "🟢 Cache HIT"
        else:
            with st.spinner("🤖 Retrieving and generating..."):
                answer, formatted_context = answer_with_openai(user_input)
                cache.set(user_input, (answer, formatted_context))
            cache_status = "🔵 Cache MISS"

        elapsed = time.time() - start

        # --- Display Assistant Response ---
        with st.chat_message("assistant"):
            st.markdown(f"### 🧠 Answer\n{answer}")
            with st.expander("📘 Supporting Evidence", expanded=False):
                st.markdown(formatted_context)

        st.caption(f"{cache_status} | ⏱ {elapsed:.2f}s")
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ---------------------- CHAT HISTORY ----------------------
if st.session_state.chat_history:
    st.divider()
    st.subheader("🕒 Conversation History")
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])
