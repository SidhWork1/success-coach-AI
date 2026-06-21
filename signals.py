"""
M6 — Signal detection.

Built with LangGraph: a multi-step graph that analyzes a finished session,
checks for recurring patterns, decides severity/urgency, and conditionally
drafts a manager notification — only when the situation is critical.

Signal storage uses Mem0 (same pattern as memory.py's factual/summary split),
since Google Sheets access is currently unavailable. Each student's signals
live under a dedicated user_id namespace: "{student_id}_signals".
"""

import os
import json
from typing import TypedDict, Optional, Literal
from datetime import datetime

import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from mem0 import MemoryClient


# ─────────────────────────────────────────────────────────────────────────────
# Mem0 client + signal storage
# ─────────────────────────────────────────────────────────────────────────────

def _get_mem0_api_key():
    try:
        if "MEM0_API_KEY" in st.secrets:
            return st.secrets["MEM0_API_KEY"]
    except Exception:
        pass
    return os.environ.get("MEM0_API_KEY")


@st.cache_resource
def _get_signal_mem0_client():
    return MemoryClient(api_key=_get_mem0_api_key())


def _signal_user_id(student_id: str) -> str:
    """Separate namespace for signals, keeps them isolated from factual/summary memory."""
    return f"{student_id}_signals"


def save_signal_to_memory(signal: dict):
    """
    Stores a finished signal in Mem0, under a dedicated namespace for this student.
    Uses infer=False so the signal is stored exactly as-is (structured JSON),
    not run through Mem0's fact-extraction (which is meant for conversational text).
    """
    if not signal:
        return

    client = _get_signal_mem0_client()
    signal_text = json.dumps(signal)

    client.add(
        messages=[{"role": "assistant", "content": signal_text}],
        user_id=_signal_user_id(signal["student_id"]),
        infer=False,
    )


def get_signals_for_student(student_id: str, limit: int = 20) -> list[dict]:
    """
    Retrieves all stored signals for a student, parsed back into dicts.
    Used by check_recurrence, and by the coach dashboard.
    """
    try:
        client = _get_signal_mem0_client()
        result = client.get_all(filters={"user_id": _signal_user_id(student_id)}, limit=limit)
        signals = []
        for m in result.get("results", []):
            try:
                signals.append(json.loads(m.get("memory", "{}")))
            except json.JSONDecodeError:
                continue  # skip anything that doesn't parse cleanly
        return signals
    except Exception as e:
        print(f"Signal retrieval error: {e}")
        return []


def get_active_alerts_for_student(student_id: str) -> list[dict]:
    """Returns this student's signals filtered to high/critical severity only."""
    signals = get_signals_for_student(student_id)
    return [s for s in signals if s.get("severity") in ("high", "critical")]


def get_all_active_alerts(student_ids: list[str]) -> list[dict]:
    """
    Scans every given student_id for high/critical signals.
    Returns a flat list of alert dicts, each including student_id so the
    coach view can show whose alert it is. Sorted: critical first, then by
    most recent timestamp.
    """
    all_alerts = []
    for sid in student_ids:
        all_alerts.extend(get_active_alerts_for_student(sid))

    severity_rank = {"critical": 0, "high": 1}
    all_alerts.sort(
        key=lambda a: (severity_rank.get(a.get("severity"), 2), a.get("timestamp", "")),
        reverse=False,
    )
    # within same severity, show most recent first
    all_alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
    all_alerts.sort(key=lambda a: severity_rank.get(a.get("severity"), 2))

    return all_alerts


# ─────────────────────────────────────────────────────────────────────────────
# Structured output schemas — guarantees the LLM returns usable JSON shapes
# ─────────────────────────────────────────────────────────────────────────────

class SessionAnalysis(BaseModel):
    has_concern: bool = Field(description="True if anything concerning came up in this session")
    concern_summary: str = Field(description="Brief description of the concern, or empty string if none")
    signal_type: str = Field(
        description="One short category label, e.g. 'emotional_distress', 'academic_risk', "
                     "'administrative_issue', 'dropout_risk', or 'none'"
    )


class SeverityDecision(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    urgency: Literal["today", "within_2_days", "within_week"]
    reasoning: str = Field(description="One sentence explaining why this severity/urgency was chosen")
    next_action: str = Field(
        description="One concise, concrete suggestion for what the coach should do next "
                     "(e.g. 'Schedule a 1:1 call today to check in on their wellbeing', "
                     "'Review their attendance trend before the next session')"
    )


class ManagerNotice(BaseModel):
    draft_message: str = Field(description="A short, professional draft message to the student's manager")


# ─────────────────────────────────────────────────────────────────────────────
# Graph state — the data that flows through and gets updated at each step
# ─────────────────────────────────────────────────────────────────────────────

class SignalState(TypedDict):
    student_id: str
    student_name: str
    manager_email: str
    messages: list[dict]              # the session transcript

    # filled in as the graph runs:
    has_concern: Optional[bool]
    concern_summary: Optional[str]
    signal_type: Optional[str]
    is_recurring: Optional[bool]
    recurrence_note: Optional[str]
    severity: Optional[str]
    urgency: Optional[str]
    manager_notice: Optional[str]
    next_action: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes — each one is a single step
# ─────────────────────────────────────────────────────────────────────────────

def analyze_session(state: SignalState) -> SignalState:
    """Step 1 — read the transcript, decide if anything concerning happened."""
    llm = ChatOpenAI(model="gpt-5.4-mini-2026-03-17", temperature=0).with_structured_output(SessionAnalysis)

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"])

    prompt = f"""Review this coaching session transcript. Identify if the student raised
anything concerning — academic struggles, emotional distress, dropout intent,
administrative issues (e.g. missing hall ticket, payment problems), or anything
else a coach should know about.

If nothing concerning came up (e.g. just a routine question), set has_concern to False.

TRANSCRIPT:
{transcript}"""

    result = llm.invoke(prompt)

    state["has_concern"] = result.has_concern
    state["concern_summary"] = result.concern_summary
    state["signal_type"] = result.signal_type
    return state


def check_recurrence(state: SignalState) -> SignalState:
    """Step 2 — compare against past signals (real, from Mem0) to detect a repeating pattern."""
    if not state["has_concern"]:
        state["is_recurring"] = False
        state["recurrence_note"] = ""
        return state

    past_signals = get_signals_for_student(state["student_id"])
    same_type_signals = [s for s in past_signals if s.get("signal_type") == state["signal_type"]]

    if same_type_signals:
        state["is_recurring"] = True
        count = len(same_type_signals)
        state["recurrence_note"] = (
            f"This is the {count + 1}th time a '{state['signal_type']}' signal "
            f"has been raised for this student."
        )
    else:
        state["is_recurring"] = False
        state["recurrence_note"] = "This is the first time this type of concern has come up."

    return state


def decide_severity_urgency(state: SignalState) -> SignalState:
    """Step 3 — assign severity + urgency, informed by the concern and recurrence."""
    if not state["has_concern"]:
        state["severity"] = "low"
        state["urgency"] = "within_week"
        state["next_action"] = ""
        return state

    llm = ChatOpenAI(model="gpt-5.4-mini-2026-03-17", temperature=0).with_structured_output(SeverityDecision)

    prompt = f"""A student coaching session surfaced this concern:

CONCERN: {state['concern_summary']}
TYPE: {state['signal_type']}
RECURRENCE: {state['recurrence_note']}

Decide the severity and urgency.

SEVERITY GUIDE:
- low: minor, routine, no real risk
- medium: worth attention but not urgent
- high: clear risk (academic, emotional, or dropout risk) needing prompt coach attention
- critical: immediate safety/wellbeing concern, or explicit dropout/refund intent,
  or anything requiring urgent escalation

SAFETY FLOOR — apply regardless of how mild the rest of the conversation seems:
If the student expresses language suggesting broader hopelessness, despair, that things
feel pointless, that they don't see things improving, or any hint of self-harm or
suicidal ideation — treat this as AT LEAST "critical", even if the explicit topic was
academic. Err toward over-flagging. A coach reviewing an unnecessary critical alert
costs little; missing a genuine one costs a lot. When in doubt, escalate.

URGENCY GUIDE:
- today: coach should act today
- within_2_days: can wait a day or two
- within_week: can be scheduled within the week

Note: recurring concerns should generally be treated as MORE severe than first-time ones,
since a repeated pattern signals the issue isn't resolving on its own."""

    result = llm.invoke(prompt)

    state["severity"] = result.severity
    state["urgency"] = result.urgency
    state["next_action"] = result.next_action
    return state


def draft_manager_notice(state: SignalState) -> SignalState:
    """Step 4 (conditional) — only runs if severity is critical."""
    llm = ChatOpenAI(model="gpt-5.4-mini-2026-03-17", temperature=0.3).with_structured_output(ManagerNotice)

    prompt = f"""Draft a short, professional notification to a student's manager about a
critical concern raised during a coaching session. Be factual, calm, and clear —
this is an internal alert, not a dramatic message.

Student: {state['student_name']}
Concern: {state['concern_summary']}
Context: {state['recurrence_note']}"""

    result = llm.invoke(prompt)
    state["manager_notice"] = result.draft_message
    return state


def skip_manager_notice(state: SignalState) -> SignalState:
    """Step 4 (conditional) — runs when severity is NOT critical, just sets empty."""
    state["manager_notice"] = ""
    return state


def route_after_severity(state: SignalState) -> str:
    """Decides which branch to take after severity is assigned."""
    if state["severity"] == "critical":
        return "draft_manager_notice"
    return "skip_manager_notice"


# ─────────────────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_graph():
    graph = StateGraph(SignalState)

    graph.add_node("analyze_session", analyze_session)
    graph.add_node("check_recurrence", check_recurrence)
    graph.add_node("decide_severity_urgency", decide_severity_urgency)
    graph.add_node("draft_manager_notice", draft_manager_notice)
    graph.add_node("skip_manager_notice", skip_manager_notice)

    graph.set_entry_point("analyze_session")
    graph.add_edge("analyze_session", "check_recurrence")
    graph.add_edge("check_recurrence", "decide_severity_urgency")

    graph.add_conditional_edges(
        "decide_severity_urgency",
        route_after_severity,
        {
            "draft_manager_notice": "draft_manager_notice",
            "skip_manager_notice": "skip_manager_notice",
        },
    )

    graph.add_edge("draft_manager_notice", END)
    graph.add_edge("skip_manager_notice", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public function — call this from app.py after a session ends
# ─────────────────────────────────────────────────────────────────────────────

def run_signal_detection(student_id: str, student_name: str, manager_email: str, messages: list[dict]) -> dict:
    """
    Runs the full signal detection graph on a finished session.
    If a real concern is found, saves it to Mem0 and returns the signal dict.
    Returns None if nothing concerning was detected.
    """
    if not messages:
        return None

    graph = build_signal_graph()

    initial_state: SignalState = {
        "student_id": student_id,
        "student_name": student_name,
        "manager_email": manager_email,
        "messages": messages,
        "has_concern": None,
        "concern_summary": None,
        "signal_type": None,
        "is_recurring": None,
        "recurrence_note": None,
        "severity": None,
        "urgency": None,
        "manager_notice": None,
        "next_action": None,
    }

    final_state = graph.invoke(initial_state)

    if not final_state["has_concern"]:
        return None  # nothing worth recording as a signal

    signal = {
        "student_id": student_id,
        "signal_type": final_state["signal_type"],
        "severity": final_state["severity"],
        "urgency": final_state["urgency"],
        "reason": final_state["concern_summary"],
        "is_recurring": final_state["is_recurring"],
        "recurrence_note": final_state["recurrence_note"],
        "next_action": final_state["next_action"],
        "manager_notice": final_state["manager_notice"],
        "timestamp": datetime.now().isoformat(),
        "actioned": False,
    }

    save_signal_to_memory(signal)

    return signal
