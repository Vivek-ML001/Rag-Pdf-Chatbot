# import os
# import hashlib
# from io import BytesIO

# import chromadb
# from dotenv import load_dotenv, find_dotenv
# from sentence_transformers import SentenceTransformer
# from groq import Groq
# from pypdf import PdfReader
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# load_dotenv(find_dotenv())


# class RAG:
#     def __init__(
#         self,
#         persist_dir: str = "data/vector_store",
#         collection_name: str = "pdf_documents",
#         llm_model: str = "openai/gpt-oss-20b",
#     ):
#         self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
#         client = chromadb.PersistentClient(path=persist_dir)
#         self.collection = client.get_or_create_collection(
#             name=collection_name, metadata={"hnsw:space": "cosine"}
#         )

#         api_key = os.getenv("GROQ_API_KEY")
#         if not api_key:
#             raise ValueError("GROQ_API_KEY not found in .env")
#         self.llm = Groq(api_key=api_key)
#         self.llm_model = llm_model

#     def count(self) -> int:
#         return self.collection.count()

#     def add_pdf(self, file_bytes: bytes, filename: str) -> int:
#         """Chunk + embed + store an uploaded PDF. Returns number of chunks added."""
#         reader = PdfReader(BytesIO(file_bytes))
#         pages = []
#         for i, page in enumerate(reader.pages):
#             text = page.extract_text() or ""
#             if text.strip():
#                 pages.append(
#                     Document(
#                         page_content=text,
#                         metadata={"page": i, "page_label": str(i + 1)},
#                     )
#                 )

#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = splitter.split_documents(pages)
#         if not chunks:
#             return 0

#         texts = [c.page_content for c in chunks]
#         embeddings = self.embedder.encode(texts).tolist()

#         file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
#         ids, metadatas = [], []
#         for i, c in enumerate(chunks):
#             ids.append(f"{file_hash}_{i}")
#             metadatas.append({
#                 "source": filename,
#                 "page_label": c.metadata["page_label"],
#                 "chunk_index": i,
#             })

#         # upsert: same PDF dobara upload karo to duplicate nahi banega
#         self.collection.upsert(
#             ids=ids,
#             embeddings=embeddings,
#             documents=texts,
#             metadatas=metadatas,
#         )
#         return len(chunks)

#     def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.3):
#         total = self.collection.count()
#         if total == 0:
#             return []

#         q_emb = self.embedder.encode([query])[0].tolist()
#         res = self.collection.query(
#             query_embeddings=[q_emb],
#             n_results=min(top_k, total),
#         )

#         hits = []
#         for i in range(len(res["ids"][0])):
#             score = 1 - res["distances"][0][i]
#             if score >= score_threshold:
#                 hits.append({
#                     "content": res["documents"][0][i],
#                     "metadata": res["metadatas"][0][i],
#                     "score": score,
#                 })
#         return hits

#     def ask(self, query: str, top_k: int = 3, score_threshold: float = 0.3):
#         hits = self.retrieve(query, top_k, score_threshold)
#         if not hits:
#             return "No relevant context found.", []

#         context = "\n\n".join(h["content"] for h in hits)
#         prompt = f"""Answer ONLY using the context below. Do not use outside knowledge.
# If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
# Write formulas in plain text, not LaTeX.

# Context:
# {context}

# Question: {query}

# Answer:"""

#         resp = self.llm.chat.completions.create(
#             model=self.llm_model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=1024,
#         )
#         return resp.choices[0].message.content, hits



# import os
# import hashlib
# from io import BytesIO

# import chromadb
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())  # langsmith import se pehle

# from sentence_transformers import SentenceTransformer
# from groq import Groq
# from pypdf import PdfReader
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# try:
#     from langsmith import traceable
# except ImportError:  # langsmith na ho to no-op decorator
#     def traceable(*args, **kwargs):
#         def decorator(fn):
#             return fn
#         return decorator


# def _get_secret(name: str):
#     val = os.getenv(name)
#     if val:
#         return val
#     try:
#         import streamlit as st
#         return st.secrets.get(name)
#     except Exception:
#         return None


# class RAG:
#     def __init__(
#         self,
#         persist_dir: str = "data/vector_store",
#         collection_name: str = "pdf_documents",
#         llm_model: str = "openai/gpt-oss-20b",
#     ):
#         self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
#         client = chromadb.PersistentClient(path=persist_dir)
#         self.collection = client.get_or_create_collection(
#             name=collection_name, metadata={"hnsw:space": "cosine"}
#         )

#         api_key = _get_secret("GROQ_API_KEY")
#         if not api_key:
#             raise ValueError("GROQ_API_KEY not found in .env or Streamlit secrets")
#         self.llm = Groq(api_key=api_key)
#         self.llm_model = llm_model

#     def count(self) -> int:
#         return self.collection.count()

#     def add_pdf(self, file_bytes: bytes, filename: str) -> int:
#         """Chunk + embed + store an uploaded PDF. Returns number of chunks added."""
#         reader = PdfReader(BytesIO(file_bytes))
#         pages = []
#         for i, page in enumerate(reader.pages):
#             text = page.extract_text() or ""
#             if text.strip():
#                 pages.append(
#                     Document(
#                         page_content=text,
#                         metadata={"page": i, "page_label": str(i + 1)},
#                     )
#                 )

#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = splitter.split_documents(pages)
#         if not chunks:
#             return 0

#         texts = [c.page_content for c in chunks]
#         embeddings = self.embedder.encode(texts).tolist()

#         file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
#         ids, metadatas = [], []
#         for i, c in enumerate(chunks):
#             ids.append(f"{file_hash}_{i}")
#             metadatas.append({
#                 "source": filename,
#                 "page_label": c.metadata["page_label"],
#                 "chunk_index": i,
#             })

#         # upsert: same PDF dobara upload karo to duplicate nahi banega
#         self.collection.upsert(
#             ids=ids,
#             embeddings=embeddings,
#             documents=texts,
#             metadatas=metadatas,
#         )
#         return len(chunks)

#     @traceable(name="retrieve", run_type="retriever")
#     def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.3):
#         total = self.collection.count()
#         if total == 0:
#             return []

#         q_emb = self.embedder.encode([query])[0].tolist()
#         res = self.collection.query(
#             query_embeddings=[q_emb],
#             n_results=min(top_k, total),
#         )

#         hits = []
#         for i in range(len(res["ids"][0])):
#             score = 1 - res["distances"][0][i]
#             if score >= score_threshold:
#                 hits.append({
#                     "page_content": res["documents"][0][i],
#                     "content": res["documents"][0][i],
#                     "metadata": res["metadatas"][0][i],
#                     "score": score,
#                 })
#         return hits

#     @traceable(name="generate", run_type="llm")
#     def generate(self, query: str, hits: list):
#         context = "\n\n".join(h["content"] for h in hits)
#         prompt = f"""Answer ONLY using the context below. Do not use outside knowledge.
# If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
# Write formulas in plain text, not LaTeX.

# Context:
# {context}

# Question: {query}

# Answer:"""

#         resp = self.llm.chat.completions.create(
#             model=self.llm_model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=1024,
#         )
#         return resp.choices[0].message.content

#     @traceable(name="rag_pipeline", run_type="chain")
#     def ask(self, query: str, top_k: int = 3, score_threshold: float = 0.3):
#         hits = self.retrieve(query, top_k, score_threshold)
#         if not hits:
#             return "No relevant context found.", []
#         answer = self.generate(query, hits)
#         return answer, hits





import os
import hashlib
from io import BytesIO

import chromadb
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # langsmith import se pehle

from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langsmith import traceable
except ImportError:  # langsmith na ho to no-op decorator
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


def _get_secret(name: str):
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


class RAG:
    def __init__(
        self,
        persist_dir: str = "data/vector_store",
        collection_name: str = "pdf_documents",
        llm_model: str = "openai/gpt-oss-20b",
    ):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        api_key = _get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env or Streamlit secrets")
        self.llm = Groq(api_key=api_key)
        self.llm_model = llm_model

    def count(self) -> int:
        return self.collection.count()

    # ---------------- Ingestion ----------------
    def add_pdf(self, file_bytes: bytes, filename: str) -> int:
        """Chunk + embed + store an uploaded PDF. Returns number of chunks added."""
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"page": i, "page_label": str(i + 1)},
                    )
                )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        if not chunks:
            return 0

        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.encode(texts).tolist()

        file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
        ids, metadatas = [], []
        for i, c in enumerate(chunks):
            ids.append(f"{file_hash}_{i}")
            metadatas.append({
                "source": filename,
                "page_label": c.metadata["page_label"],
                "chunk_index": i,
            })

        # upsert: same PDF dobara upload karo to duplicate nahi banega
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    # ---------------- LLM helper ----------------
    def _complete(self, messages, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        resp = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    # ---------------- Router ----------------
    @traceable(name="route", run_type="chain")
    def route(self, query: str) -> str:
        """Returns 'chat' or 'rag'."""
        if self.collection.count() == 0:
            return "chat"

        prompt = (
            "Classify the user message. Reply with exactly ONE word: chat or rag.\n"
            "chat = greeting, thanks, small talk, or a general question that needs no document.\n"
            "rag = a question that needs information from the uploaded documents.\n\n"
            "Examples:\n"
            "hello -> chat\n"
            "thanks! -> chat\n"
            "what is the capital of France? -> chat\n"
            "summarize the document -> rag\n"
            "what does the paper say about attention? -> rag\n"
            "explain section 3 -> rag\n\n"
            f"Message: {query}\nAnswer:"
        )
        try:
            out = self._complete(
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,  # reasoning model hai, chhota limit rakha to output khali aa sakta hai
            ).lower()
        except Exception:
            return "rag"  # router fail ho to document mode safe hai

        return "chat" if "chat" in out and "rag" not in out else "rag"

    # ---------------- Retrieval ----------------
    @traceable(name="retrieve", run_type="retriever")
    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.3):
        total = self.collection.count()
        if total == 0:
            return []

        q_emb = self.embedder.encode([query])[0].tolist()
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, total),
        )

        hits = []
        for i in range(len(res["ids"][0])):
            score = 1 - res["distances"][0][i]
            if score >= score_threshold:
                hits.append({
                    "page_content": res["documents"][0][i],
                    "content": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": score,
                })
        return hits

    # ---------------- Generation ----------------
    @traceable(name="generate", run_type="llm")
    def generate(self, query: str, hits: list):
        context = "\n\n".join(h["content"] for h in hits)
        prompt = f"""Answer ONLY using the context below. Do not use outside knowledge.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
Write formulas in plain text, not LaTeX.

Context:
{context}

Question: {query}

Answer:"""
        return self._complete([{"role": "user", "content": prompt}], temperature=0.1)

    @traceable(name="chat", run_type="llm")
    def chat(self, query: str, history: list | None = None) -> str:
        messages = [{
            "role": "system",
            "content": (
                "You are a friendly assistant inside a PDF chat app. "
                "Answer greetings and general questions normally. "
                "When relevant, mention that the user can upload PDFs and ask questions about them."
            ),
        }]
        if history:
            messages += history[-6:]  # [{"role": "user"/"assistant", "content": "..."}]
        messages.append({"role": "user", "content": query})
        return self._complete(messages, temperature=0.5)

    # ---------------- Main entry ----------------
    @traceable(name="rag_pipeline", run_type="chain")
    def ask(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3,
        history: list | None = None,
    ):
        if self.route(query) == "chat":
            return self.chat(query, history), []

        hits = self.retrieve(query, top_k, score_threshold)
        if not hits:
            return "Ye information uploaded documents mein nahi mili.", []

        return self.generate(query, hits), hits
