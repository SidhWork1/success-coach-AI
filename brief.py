"""
M8 — Pre-meeting student brief.

Pulls together everything a coach needs before talking to a specific student:
academic data, factual memory, session summaries, and active alerts.
Synthesizes it into a focused, conversational brief via one LLM call.
"""

from langchain_openai import ChatOpenAI
from sheets import fetch_student_data, verify_student_identity
from memory import get_factual_memory, get_summary_memory
from signals import get_active_alerts_for_student


def build_student_brief(student_id: str) -> dict:
    """
    Returns:
        {
            "found": bool,
            "student_name": str,
            "brief_text": str,   # the synthesized brief, ready to display
        }
        or {"found": False} if the student doesn't exist.
    """
    roster_row = verify_student_identity(student_id)
    if roster_row is None:
        return {"found": False}

    student_name = roster_row.get("name", student_id)

    academic_data = fetch_student_data(student_id)
    factual = get_factual_memory(student_id, limit=15)
    summaries = get_summary_memory(student_id, limit=10)
    active_alerts = get_active_alerts_for_student(student_id)

    # Format each piece for the prompt
    scores_text = "\n".join(
        f"- {s['subject']}: {s['score']}/{s['max_score']}" for s in academic_data.get("scores", [])
    ) or "No scores recorded."

    if academic_data.get("attendance"):
        latest = sorted(academic_data["attendance"], key=lambda r: r["week_of"])[-1]
        attendance_text = f"Most recent week ({latest['week_of']}): {latest['attendance_pct']}%"
    else:
        attendance_text = "No attendance data."

    exams_text = "\n".join(
        f"- {e['subject']} ({e['exam_type']}) on {e['exam_date']}"
        for e in academic_data.get("exam_schedule", [])
    ) or "No upcoming exams recorded."

    factual_text = "\n".join(f"- {f}" for f in factual) or "No factual history yet."

    # Summaries are stored newest-last typically; show most recent few, oldest to newest
    summaries_text = "\n".join(f"- {s}" for s in summaries) or "No past session summaries yet."

    alerts_text = "\n".join(
        f"- [{a['severity'].upper()}] {a['reason']}" for a in active_alerts
    ) or "No active alerts."

    llm = ChatOpenAI(model="gpt-5.4-mini-2026-03-17", temperature=0.4)

    prompt = f"""You are preparing a coach for a 1:1 meeting with a student.
Write a focused, conversational pre-meeting brief — NOT a formal report.
Use short paragraphs or brief bullet points. Be direct and useful, not padded.

STUDENT: {student_name} ({student_id})

ACADEMIC SCORES:
{scores_text}

ATTENDANCE:
{attendance_text}

UPCOMING EXAMS:
{exams_text}

WHAT WE KNOW ABOUT THIS STUDENT (factual history across sessions):
{factual_text}

PAST SESSION SUMMARIES (most recent sessions, in order):
{summaries_text}

ACTIVE ALERTS:
{alerts_text}

Write the brief covering exactly these four things, in this order:
1. **Current academic situation** — one or two sentences, the real picture
2. **What's changed since the last session** — compare recent vs older summaries if multiple exist; say "first session" if there's only one or none
3. **Open concerns** — anything unresolved, recurring, or flagged
4. **Conversation starters** — 2-3 specific, natural things the coach could open with today, grounded in what's actually known about this student (not generic)

Keep the whole brief under 200 words. Write it for a coach about to walk into the room in 2 minutes."""

    response = llm.invoke(prompt)

    return {
        "found": True,
        "student_name": student_name,
        "brief_text": response.content,
    }