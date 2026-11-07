import uuid
import streamlit as st
from src.rag.rag_pipeline import RAG

st.set_page_config(page_title="RAG Chat", page_icon="📄", layout="wide")


@st.cache_resource
def load_rag():
    return RAG()


rag = load_rag()

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 2rem;}
    .header {text-align: center; margin-bottom: 1.5rem;}
    .title {font-size: 2.4rem; font-weight: 700;}
    .title span {background: linear-gradient(90deg,#4f8cff,#a855f7);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {color: #888; margin-top: .2rem;}
    .src-card {border: 1px solid rgba(128,128,128,.3); border-radius: 10px;
               padding: .5rem .8rem; margin-bottom: .4rem; font-size: .85rem;}
    .score {float: right; font-weight: 600; color: #4f8cff;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------- Chat state ----------
def new_chat():
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "New chat", "messages": []}
    st.session_state.current = cid


if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_chat()

chat = st.session_state.chats[st.session_state.current]

# ---------- Sidebar ----------
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        # khali chat already open ho to naya mat banao
        if chat["messages"]:
            new_chat()
        st.rerun()

    st.markdown("### 💬 Chats")
    for cid in reversed(list(st.session_state.chats.keys())):
        c = st.session_state.chats[cid]
        label = ("🟢 " if cid == st.session_state.current else "") + c["title"][:28]
        col1, col2 = st.columns([5, 1])
        if col1.button(label, key=f"open_{cid}", use_container_width=True):
            st.session_state.current = cid
            st.rerun()
        if col2.button("✕", key=f"del_{cid}"):
            del st.session_state.chats[cid]
            if not st.session_state.chats:
                new_chat()
            elif st.session_state.current == cid:
                st.session_state.current = list(st.session_state.chats.keys())[-1]
            st.rerun()

    st.divider()
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top K chunks", 1, 10, 6)
    threshold = st.slider("Score threshold", 0.0, 1.0, 0.30, 0.05)

    st.divider()
    st.markdown("### 📤 Upload PDF")
    files = st.file_uploader("Add PDFs to knowledge base", type="pdf",
                             accept_multiple_files=True)
    if st.button("Add to knowledge base", use_container_width=True) and files:
        with st.spinner("Indexing..."):
            for f in files:
                n = rag.add_pdf(f.getvalue(), f.name)
                st.success(f"{f.name}: {n} chunks")

    st.divider()
    st.metric("Chunks in DB", rag.count())

# ---------- Header ----------
st.markdown(
    """
    <div class="header">
        <div class="title">📄 <span>RAG Chat with Groq</span></div>
        <div class="subtitle">Ask questions about your documents</div>
    </div>
    """,
    unsafe_allow_html=True,
)

def show_sources(hits):
    if not hits:
        return
    with st.expander(f"📚 Sources ({len(hits)})"):
        for h in hits:
            m = h["metadata"]
            st.markdown(
                f'<div class="src-card"><b>{m["source"]}</b> · Page {m["page_label"]}'
                f'<span class="score">{h["score"]:.2f}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(h["content"][:300] + "...")


# ---------- Show current chat ----------
if not chat["messages"]:
    st.markdown(
        "<p style='text-align:center;color:#888'>Upload a PDF from the sidebar and start asking questions 👇</p>",
        unsafe_allow_html=True,
    )

    
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("hits"))

# ---------- Input ----------
if query := st.chat_input("Ask a question about your documents"):
    if not chat["messages"]:
        chat["title"] = query[:40]  # pehle question se title

    chat["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, hits = rag.ask(query, top_k, threshold)
        st.markdown(answer)
        show_sources(hits)

    chat["messages"].append({"role": "assistant", "content": answer, "hits": hits})
    st.rerun()  # sidebar mein naya title turant dikhe
