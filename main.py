import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sheets import fetch_student_data, verify_student_identity
import json
from datetime import datetime

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini-2026-03-17", temperature=0.7)

st.title("🎓 Success Coach AI")


# ── helper: build a system prompt from list-based student data ────────────────

def build_system_prompt(student_info: dict, student_data: dict) -> str:
    name    = student_info.get("name", "")
    program = student_info.get("program", "")
    cohort  = student_info.get("cohort", "")

    scores     = student_data.get("scores", [])
    attendance = student_data.get("attendance", [])
    exams      = student_data.get("exam_schedule", [])

    # ── compute a few derived facts so the LLM doesn't have to guess ──
    flags = []

    if scores:
        avg_score = sum(float(r["score"]) for r in scores) / len(scores)
        low_subjects = [r["subject"] for r in scores if float(r["score"]) < 50]
        if low_subjects:
            flags.append(f"Low scores (<50) in: {', '.join(low_subjects)}")
    else:
        avg_score = None

    if attendance:
        latest_week = sorted(attendance, key=lambda r: r["week_of"])[-1]
        latest_pct  = float(latest_week["attendance_pct"])
        if latest_pct < 75:
            flags.append(f"Latest week attendance is low: {latest_pct}% (week of {latest_week['week_of']})")
    else:
        latest_pct = None

    if exams:
        today = datetime.now().date()
        upcoming = sorted(exams, key=lambda r: r["exam_date"])
        for e in upcoming:
            try:
                exam_date = datetime.strptime(e["exam_date"], "%Y-%m-%d").date()
                days_away = (exam_date - today).days
                if 0 <= days_away <= 7:
                    flags.append(f"Exam '{e['subject']}' ({e['exam_type']}) is in {days_away} day(s) on {e['exam_date']}")
            except ValueError:
                pass

    flags_text = "\n".join(f"- {f}" for f in flags) if flags else "- None right now"

    prompt = f"""You are a warm, supportive academic success coach AI.

You are speaking with {name} (Program: {program}, Cohort: {cohort}).

EXAM SCORES (all subjects so far):
{json.dumps(scores, indent=2)}
{f"Average score: {avg_score:.1f}/100" if avg_score is not None else "No scores yet."}

ATTENDANCE (weekly):
{json.dumps(attendance, indent=2)}
{f"Most recent week attendance: {latest_pct}%" if latest_pct is not None else "No attendance data yet."}

UPCOMING EXAMS:
{json.dumps(exams, indent=2)}

AUTOMATICALLY DETECTED CONCERNS:
{flags_text}

Your job:
- Answer questions using this data. Use the exact numbers given — never estimate or invent.
- If there are detected concerns above, proactively mention them when relevant, 
  especially if the student seems unaware or asks a related question.
- Be encouraging but honest.
- If asked about something not present in the data, say you don't have that info yet.
- Keep responses conversational, not like a formal report.
"""
    return prompt


# ── session state init ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "student_info" not in st.session_state:
    st.session_state.student_info = None
if "student_data" not in st.session_state:
    st.session_state.student_data = None
if "identifying" not in st.session_state:
    st.session_state.identifying = "ask_id"


# ── identification flow ─────────────────────────────────────────────────────────

if st.session_state.identifying != "done":

    if st.session_state.identifying == "ask_id":
        with st.chat_message("assistant"):
            st.markdown(
                "Hi there! 👋 I'm your Success Coach. "
                "Could you share your **Student ID** so I can pull up your profile?"
            )

        id_input = st.chat_input("Your Student ID (e.g. STU001)...")
        if id_input:
            with st.spinner("Looking you up..."):
                roster_row = verify_student_identity(id_input.strip())

            if roster_row is None:
                st.error(
                    f"I couldn't find **{id_input.strip()}** in the system. "
                    "Please check your Student ID and try again."
                )
            else:
                st.session_state.student_info = {
                    "student_id":    roster_row.get("student_id", ""),
                    "name":          roster_row.get("name", ""),
                    "program":       roster_row.get("program", ""),
                    "cohort":        roster_row.get("cohort", ""),
                    "manager_email": roster_row.get("manager_email", ""),
                }

                with st.spinner("Fetching your academic data..."):
                    data = fetch_student_data(id_input.strip())

                st.session_state.student_data = data
                st.session_state.identifying  = "done"

                name    = st.session_state.student_info["name"]
                program = st.session_state.student_info["program"]
                greeting = (
                    f"Welcome, **{name}**! 👋 I can see you're in the "
                    f"**{program}** program. I've pulled up your scores, "
                    "attendance and exam schedule. Ask me anything!"
                )
                st.session_state.messages.append({"role": "assistant", "content": greeting})
                st.rerun()

    st.stop()


# ── main chat ─────────────────────────────────────────────────────────────────

system_prompt = build_system_prompt(st.session_state.student_info, st.session_state.student_data)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    messages_for_llm = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    response = llm.invoke(messages_for_llm)
    answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})