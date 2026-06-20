"""
Mem0 integration — stores and retrieves two distinct memory types per student,
separated using a user_id suffix instead of metadata (metadata tagging has a
known reliability issue with Mem0's infer=True extraction pipeline).
"""

import os
import streamlit as st
from mem0 import MemoryClient
from langchain_openai import ChatOpenAI


def _get_mem0_api_key():
    try:
        if "MEM0_API_KEY" in st.secrets:
            return st.secrets["MEM0_API_KEY"]
    except Exception:
        pass
    return os.environ.get("MEM0_API_KEY")


@st.cache_resource
def get_mem0_client():
    return MemoryClient(api_key=_get_mem0_api_key())


def _summary_user_id(student_id: str) -> str:
    """Separate namespace for summary memories, keeps them isolated from factual."""
    return f"{student_id}_summary"


# ─────────────────────────────────────────────────────────────────────────────
# SAVING
# ─────────────────────────────────────────────────────────────────────────────

def _generate_session_summary(messages: list[dict]) -> str:
    llm = ChatOpenAI(model="gpt-5.4-mini-2026-03-17", temperature=0.3)
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    prompt = f"""Summarize this coaching session in 2-4 sentences.
Focus on: what the student brought up, what was discussed, and what (if anything) was decided or recommended.
Do not include pleasantries or filler. Be factual and concise.

CONVERSATION:
{transcript}

SUMMARY:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def save_session_to_memory(student_id: str, messages: list[dict]):
    if not messages:
        return

    client = get_mem0_client()

    # FACTUAL — normal user_id, Mem0 auto-extracts from raw conversation
    client.add(messages=messages, user_id=student_id)

    # SUMMARY — separate user_id namespace, our own generated summary text
    summary_text = _generate_session_summary(messages)
    client.add(
        messages=[{"role": "assistant", "content": summary_text}],
        user_id=_summary_user_id(student_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVING
# ─────────────────────────────────────────────────────────────────────────────

def get_factual_memory(student_id: str, limit: int = 10) -> list[str]:
    try:
        client = get_mem0_client()
        result = client.get_all(filters={"user_id": student_id}, limit=limit)
        return [m.get("memory", "") for m in result.get("results", [])]
    except Exception as e:
        print(f"Mem0 factual retrieval error: {e}")
        return []


def get_summary_memory(student_id: str, limit: int = 20) -> list[str]:
    try:
        client = get_mem0_client()
        result = client.get_all(filters={"user_id": _summary_user_id(student_id)}, limit=limit)
        return [m.get("memory", "") for m in result.get("results", [])]
    except Exception as e:
        print(f"Mem0 summary retrieval error: {e}")
        return []