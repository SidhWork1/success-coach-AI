import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sheets import fetch_student_data, verify_student_identity, get_all_students
from rag import retrieve_relevant_chunks
from memory import save_session_to_memory, get_factual_memory
from signals import run_signal_detection, get_all_active_alerts
import json
from datetime import datetime

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini-2026-03-17", temperature=0.7)

st.set_page_config(page_title="Success Coach AI", page_icon="🎓")


# ── helper: build a system prompt from list-based student data ────────────────

def build_system_prompt(student_info: dict, student_data: dict, retrieved_chunks: list[str], factual_memories: list[str]) -> str:
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

    if retrieved_chunks:
        kb_text = "\n\n---\n\n".join(retrieved_chunks)
    else:
        kb_text = "No specific reference material found for this question."

    if factual_memories:
        memory_text = "\n".join(f"- {m}" for m in factual_memories)
    else:
        memory_text = "No history yet — this is likely their first session."

    prompt = f"""You are a warm, supportive academic success coach AI.

You are speaking with {name} (Program: {program}, Cohort: {cohort}).

WHAT YOU KNOW ABOUT THIS STUDENT FROM PAST SESSIONS:
{memory_text}

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

REFERENCE MATERIAL (retrieved from the learning portal guide — may or may not be relevant to this specific question):
{kb_text}

Your job:
- Use what you know about this student's history naturally — don't just list facts back at them,
  let it shape your tone and what you check in on. For example, if they've struggled with
  something before, notice if it comes up again. If something helped them before, you can
  suggest it again if relevant.
- Answer questions using the student data above when relevant.
- For questions about using the learning portal, logging in, accessing schedules,
  raising doubts, or other platform/process topics — use the REFERENCE MATERIAL above.
  Only use it if it's actually relevant; ignore it otherwise.
- For general academic/conceptual questions (e.g. "what is supervised learning") —
  answer from your own knowledge.
- If neither the data nor reference material covers what's asked, say so honestly.
- Be encouraging but honest. Never make up data or claim memories you don't actually have above.
- Keep responses conversational, not like a formal report.
"""
    return prompt


# ── top-level session state ────────────────────────────────────────────────────

if "role" not in st.session_state:
    st.session_state.role = None  # None | "student" | "coach"


# ── landing screen: role selection ──────────────────────────────────────────────

if st.session_state.role is None:
    st.title("🎓 Success Coach AI")
    st.markdown("Welcome! Who are you?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎓 I'm a Student", use_container_width=True):
            st.session_state.role = "student"
            st.rerun()
    with col2:
        if st.button("🧑‍🏫 I'm a Coach", use_container_width=True):
            st.session_state.role = "coach"
            st.rerun()

    st.stop()


# ── allow switching role at any time, from the sidebar ──────────────────────────

if st.sidebar.button("← Switch role"):
    st.session_state.role = None
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT FLOW
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.role == "student":

    st.title("🎓 Success Coach AI")

    # ── student-specific session state init ──────────────────────────────────

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "student_info" not in st.session_state:
        st.session_state.student_info = None
    if "student_data" not in st.session_state:
        st.session_state.student_data = None
    if "identifying" not in st.session_state:
        st.session_state.identifying = "ask_id"

    # ── identification flow ──────────────────────────────────────────────────

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

    # ── sidebar: student info + End Session ──────────────────────────────────

    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.student_info.get('name', '')}")
        st.markdown(f"**Student ID:** {st.session_state.student_info.get('student_id', '')}")

        if st.button("🔚 End Session"):
            with st.spinner("Saving session..."):
                save_session_to_memory(
                    student_id=st.session_state.student_info["student_id"],
                    messages=st.session_state.messages,
                )

            with st.spinner("Checking in on how that session went..."):
                signal = run_signal_detection(
                    student_id=st.session_state.student_info["student_id"],
                    student_name=st.session_state.student_info["name"],
                    manager_email=st.session_state.student_info.get("manager_email", ""),
                    messages=st.session_state.messages,
                )

            # TEMPORARY: signal_sheet write is blocked until Google account access
            # is restored. For now, just display what WOULD be written, so the
            # full pipeline is visible and testable end-to-end.
            if signal:
                st.warning("⚠️ A signal was detected this session (placeholder — not yet written to signal_sheet):")
                st.json(signal)
            else:
                st.caption("No concerning signal detected this session.")

            st.session_state.messages = []
            st.session_state.student_info = None
            st.session_state.student_data = None
            st.session_state.identifying = "ask_id"

            st.success("Session saved. See you next time!")
            st.button("Continue")  # gives the user a moment to read the signal before it clears on rerun

    # ── main chat ─────────────────────────────────────────────────────────────

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me anything...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            retrieved_chunks = retrieve_relevant_chunks(prompt)
            factual_memories = get_factual_memory(st.session_state.student_info["student_id"])

        system_prompt = build_system_prompt(
            st.session_state.student_info,
            st.session_state.student_data,
            retrieved_chunks,
            factual_memories,
        )

        messages_for_llm = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ]

        response = llm.invoke(messages_for_llm)
        answer = response.content

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ═══════════════════════════════════════════════════════════════════════════════
# COACH FLOW
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.role == "coach":

    st.title("🧑‍🏫 Coach View")

    with st.spinner("Checking for flagged students..."):
        all_students = get_all_students()
        student_ids = [s["student_id"] for s in all_students if s.get("student_id")]
        student_lookup = {s["student_id"]: s for s in all_students}
        alerts = get_all_active_alerts(student_ids)

    if not alerts:
        st.success("✅ No high or critical alerts right now. All clear.")
    else:
        st.warning(f"⚠️ {len(alerts)} active alert(s) need attention.")

        for alert in alerts:
            student = student_lookup.get(alert["student_id"], {})
            student_name = student.get("name", alert["student_id"])

            severity_icon = "🔴" if alert["severity"] == "critical" else "🟠"

            with st.container(border=True):
                st.markdown(f"### {severity_icon} {student_name} — {alert['severity'].upper()}")
                st.caption(f"Urgency: {alert['urgency'].replace('_', ' ')} · Type: {alert['signal_type']}")

                st.markdown(f"**Situation:** {alert['reason']}")
                st.markdown(f"**Pattern:** {alert['recurrence_note']}")

                if alert.get("next_action"):
                    st.markdown(f"**Suggested next step:** {alert['next_action']}")

                if alert["severity"] == "critical" and alert.get("manager_notice"):
                    with st.expander("📧 Draft notification for manager"):
                        st.text_area(
                            "Draft message",
                            value=alert["manager_notice"],
                            height=100,
                            disabled=True,
                            key=f"manager_notice_{alert['student_id']}_{alert['timestamp']}",
                        )

    st.divider()
    st.caption("Ask the AI coach assistant (coming soon)")