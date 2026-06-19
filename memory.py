"""
Mem0 integration — stores session conversations persistently per student.
"""

import os
import streamlit as st
from mem0 import MemoryClient


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


def save_session_to_memory(student_id: str, messages: list[dict]):
    """
    Sends the full session's messages to Mem0, tagged with student_id.
    Mem0 automatically extracts relevant facts from the raw conversation.

    messages: list of {"role": "user"/"assistant", "content": "..."}
    """
    if not messages:
        return  # nothing to save if they didn't actually chat

    client = get_mem0_client()
    client.add(
        messages=messages,
        user_id=student_id,
    )