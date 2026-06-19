"""
ONE-TIME SETUP SCRIPT.
Run this manually whenever SETUP_GUIDE 3.md changes:
    python ingest.py

What it does:
1. Reads the markdown file
2. Splits it into overlapping chunks
3. Sends each chunk to OpenAI to get its embedding (a list of numbers
   representing the chunk's meaning)
4. Saves chunks + embeddings into a local folder called chroma_db/

After this runs, app.py just READS from chroma_db/ — it never re-does this work.
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

SOURCE_FILE   = "SETUP_GUIDE 3.md"
CHROMA_PATH   = "chroma_db"
COLLECTION    = "course_knowledge_base"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100


def main():
    # 1. Read the markdown file
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Loaded {SOURCE_FILE} — {len(raw_text)} characters total")

    # 2. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph breaks first, falls back to sentences/words
    )
    chunks = splitter.split_text(raw_text)
    print(f"Split into {len(chunks)} chunks")

    # 3. Set up OpenAI embedding function (uses OPENAI_API_KEY from .env)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    # 4. Create/connect to a persistent ChromaDB folder
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # If the collection already exists from a previous run, delete it first
    # so we don't end up with duplicate/stale chunks after re-ingesting.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"Removed old '{COLLECTION}' collection (re-ingesting fresh)")

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=openai_ef,
    )

    # 5. Add all chunks — Chroma will embed them automatically using openai_ef
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        ids=ids,
    )

    print(f"Done. {len(chunks)} chunks embedded and saved to '{CHROMA_PATH}/'")


if __name__ == "__main__":
    main()