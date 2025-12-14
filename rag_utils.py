import psycopg2
import numpy as np
import os
from langchain_community.embeddings import InfinityEmbeddings

def get_connection():
    return psycopg2.connect(
        dbname="aidb",
        user="aiuser",
        password="aipassword",
        host="localhost",
        port="5432"
    )

# Initialize the embedding model
INFINITY_API_URL=os.getenv("INFINITY_API_URL", "http://localhost:8080")

dense_model = InfinityEmbeddings(model="", infinity_api_url=INFINITY_API_URL)

def get_embedding(text):
    try:
        # Returns a list of floats
        return dense_model.embed_query(text)
    except Exception as e:
        print("Embedding error:", e)
        return np.random.rand(768).tolist()

def get_context(query, top_k=3):
    conn = get_connection()
    cur = conn.cursor()

    query_vec = get_embedding(query)
    # Format as pgvector string
    vec_str = f"[{','.join(str(x) for x in query_vec)}]"

    cur.execute("""
        SELECT content
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """, (vec_str, top_k))

    rows = cur.fetchall()
    conn.close()
    return "\n".join(r[0] for r in rows) if rows else "No context found."

def save_message(role, content):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (role, content) VALUES (%s, %s)",
        (role, content)
    )
    conn.commit()
    conn.close()

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
