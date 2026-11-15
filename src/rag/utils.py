import os
import sqlite3

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_top_chunk(chunks):
    if not chunks:
        return None
    return chunks[0]["text"]


def average_score(scores):
    return sum(scores) / len(scores) if scores else 0


def add_to_history(message, history=None):
    if history is None:
        history = []
    history.append(message)
    return history


def load_pdf_text(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None


def find_user(name):
    conn = sqlite3.connect("chat.db")
    return conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchall()