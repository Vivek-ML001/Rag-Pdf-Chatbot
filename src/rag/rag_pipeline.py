import os
import hashlib
from io import BytesIO

import chromadb
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())


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

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        self.llm = Groq(api_key=api_key)
        self.llm_model = llm_model

    def count(self) -> int:
        return self.collection.count()

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
                    "content": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": score,
                })
        return hits

    def ask(self, query: str, top_k: int = 3, score_threshold: float = 0.3):
        hits = self.retrieve(query, top_k, score_threshold)
        if not hits:
            return "No relevant context found.", []

        context = "\n\n".join(h["content"] for h in hits)
        prompt = f"""Answer ONLY using the context below. Do not use outside knowledge.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
Write formulas in plain text, not LaTeX.

Context:
{context}

Question: {query}

Answer:"""

        resp = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return resp.choices[0].message.content, hits