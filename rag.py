"""
Retrieval logic — used on every chat message.
Given a student's question, finds the most relevant chunks from chroma_db/.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

CHROMA_PATH = "chroma_db"
COLLECTION  = "course_knowledge_base"
TOP_K       = 4


def _get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")


@st.cache_resource
def get_collection():
    """
    Connects to the existing chroma_db/ folder (created by ingest.py).
    Cached so it only connects once per app session, not on every message.
    """
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=_get_openai_api_key(),
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION, embedding_function=openai_ef)


def retrieve_relevant_chunks(question: str, top_k: int = TOP_K) -> list[str]:
    """
    Given a student's question, returns the top_k most relevant
    chunks of text from the knowledge base.
    Returns an empty list if the collection is empty or something goes wrong.
    """
    try:
        collection = get_collection()
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
        )
        return results["documents"][0]  # list of chunk strings
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return []