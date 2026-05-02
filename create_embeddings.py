import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
print("URL:", SUPABASE_URL)
print("KEY:", SUPABASE_KEY)

# -----------------------------
# INIT CLIENTS
# -----------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_KEY)


# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text: str, chunk_size=500, chunk_overlap=100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


# -----------------------------
# EMBEDDING
# -----------------------------
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text).tolist()


# -----------------------------
# FETCH CHAPTERS
# -----------------------------
def fetch_chapters():
    res = supabase.table("chapters").select("id, content").execute()
    return res.data


# -----------------------------
# STORE EMBEDDINGS
# -----------------------------
def store_embeddings(chapter_id: str, chunks: List[str]):
    records = []

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        records.append({
            "chapter_id": chapter_id,
            "chunk_index": i,
            "content_chunk": chunk,
            "embedding": embedding
        })

    supabase.table("chapter_embeddings").insert(records).execute()


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run():
    chapters = fetch_chapters()

    print(f"Found {len(chapters)} chapters")

    for chapter in chapters:
        chapter_id = chapter["id"]
        content = chapter["content"]

        print(f"Processing chapter: {chapter_id}")

        chunks = chunk_text(content)

        print(f"→ {len(chunks)} chunks created")

        store_embeddings(chapter_id, chunks)

        print("→ Stored embeddings\n")


if __name__ == "__main__":
    run()