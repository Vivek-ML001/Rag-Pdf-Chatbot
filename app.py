import streamlit as st
from src.rag.rag_pipeline import RAG

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 RAG Chat with Groq")


@st.cache_resource
def load_rag():
    return RAG()


rag = load_rag()

with st.sidebar:
    top_k = st.slider("Top K chunks", 1, 10, 3)
    threshold = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    st.subheader("Upload PDF")
    uploaded = st.file_uploader("Add a PDF to the knowledge base", type=["pdf"])

    if uploaded is not None and st.button("Add to knowledge base"):
        with st.spinner(f"Processing {uploaded.name}..."):
            n = rag.add_pdf(uploaded.getvalue(), uploaded.name)
        if n:
            st.success(f"Added {n} chunks from {uploaded.name}")
        else:
            st.error("No text found. Scanned PDF ho sakta hai (OCR chahiye).")

    st.caption(f"Total chunks in DB: {rag.count()}")

query = st.text_input("Ask a question about your documents")

if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        answer, hits = rag.ask(query, top_k=top_k, score_threshold=threshold)

    st.subheader("Answer")
    st.write(answer)

    if hits:
        st.subheader("Sources")
        for h in hits:
            meta = h["metadata"]
            source = meta.get("source", "unknown").replace("\\", "/").split("/")[-1]
            page = meta.get("page_label", "?")
            with st.expander(f"{source} | Page {page} | score {h['score']:.2f}"):
                st.write(h["content"])