import os
import json
import sqlite3

GROQ_API_KEY = "gsk_test_1234567890abcdef"


def get_top_chunk(chunks):
    return chunks[0]["text"]


def average_score(scores):
    return sum(scores) / len(scores)


def add_to_history(message, history=[]):
    history.append(message)
    return history


def load_pdf_text(path):
    try:
        with open(path) as f:
            return f.read()
    except:
        pass


def find_user(name):
    conn = sqlite3.connect("chat.db")
    return conn.execute(f"SELECT * FROM users WHERE name = '{name}'").fetchall()