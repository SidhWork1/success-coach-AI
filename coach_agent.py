"""
Coach chat agent — a LangChain tool-calling agent that lets the coach ask
for things in natural language ("brief me on STU001", "who's flagged today",
"build my day, I have 5 hours") and routes to the right underlying function.

Each existing piece of functionality (M6 alerts, M7 plan, M8 brief) is wrapped
as a @tool. The agent decides which tool(s) to call based on what the coach asks.

NOTE: Uses langchain.agents.create_agent (LangChain 1.0+ API). The older
create_tool_calling_agent + AgentExecutor pattern is deprecated/removed.
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from sheets import get_all_students
from signals import get_all_active_alerts
from plan import build_daily_plan, dedupe_alerts_by_student
from brief import build_student_brief


# ─────────────────────────────────────────────────────────────────────────────
# Tools — thin wrappers around existing functions, with clear descriptions
# so the LLM knows when to use each one
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_flagged_students() -> str:
    """Use this when the coach asks who is flagged, who needs attention, who's
    at risk, or wants to see today's alerts. Returns a list of students with
    active high/critical signals, including severity and reason."""
    all_students = get_all_students()
    student_ids = [s["student_id"] for s in all_students if s.get("student_id")]
    student_lookup = {s["student_id"]: s for s in all_students}

    alerts = get_all_active_alerts(student_ids)
    alerts = dedupe_alerts_by_student(alerts)

    if not alerts:
        return "No students are currently flagged. All clear."

    lines = []
    for a in alerts:
        name = student_lookup.get(a["student_id"], {}).get("name", a["student_id"])
        lines.append(
            f"- {name} ({a['student_id']}): {a['severity'].upper()} severity, "
            f"urgency: {a['urgency']}. Reason: {a['reason']}. "
            f"Suggested action: {a.get('next_action', 'N/A')}"
        )
    return "\n".join(lines)


@tool
def get_today_plan(available_hours: float) -> str:
    """Use this when the coach wants to build, generate, or see today's schedule
    or plan, given a number of available hours. Always ask the coach for their
    available hours if they haven't stated it, before calling this tool."""
    all_students = get_all_students()
    student_ids = [s["student_id"] for s in all_students if s.get("student_id")]
    student_lookup = {s["student_id"]: s for s in all_students}

    plan = build_daily_plan(available_hours, student_ids, student_lookup)

    lines = [f"Plan using {plan['total_minutes_used']} of {plan['total_minutes_available']} available minutes."]

    lines.append("\nTODAY'S SCHEDULE:")
    if not plan["today"]:
        lines.append("Nothing scheduled — no active alerts or no time available.")
    else:
        for i, item in enumerate(plan["today"], start=1):
            lines.append(
                f"{i}. {item['name']} — {item['session_type']} ({item['duration_minutes']} min). "
                f"Why: {item['reason']}"
            )

    lines.append("\nDEFERRED TO TOMORROW:")
    if not plan["deferred"]:
        lines.append("Nobody deferred — everyone fit into today.")
    else:
        for item in plan["deferred"]:
            lines.append(f"- {item['name']}: {item['reason']}")

    return "\n".join(lines)


@tool
def get_student_brief(student_id: str) -> str:
    """Use this when the coach asks for a brief, summary, or background on a
    SPECIFIC named student before a meeting. Requires the student's ID
    (e.g. STU001). If the coach gives a name instead of an ID, ask them for
    the ID, or look it up using get_flagged_students or roster context if available."""
    result = build_student_brief(student_id)
    if not result["found"]:
        return f"No student found with ID '{student_id}'. Please check the ID and try again."
    return f"Brief for {result['student_name']} ({student_id}):\n\n{result['brief_text']}"


TOOLS = [get_flagged_students, get_today_plan, get_student_brief]

SYSTEM_PROMPT = (
    "You are a helpful assistant for a student success coach. You have tools "
    "to check flagged students, build today's schedule, and get briefs on "
    "specific students. Use the tools whenever the coach's request matches "
    "what they do — don't try to answer from memory. Be concise and direct "
    "in your responses; the coach is busy."
)


# ─────────────────────────────────────────────────────────────────────────────
# Agent setup
# ─────────────────────────────────────────────────────────────────────────────

def build_coach_agent():
    """Builds the agent once. Cheap to call, but app.py caches this in session_state."""
    agent = create_agent(
        model="gpt-5.4-mini-2026-03-17",
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def run_coach_agent(agent, user_input: str, chat_history: list[dict]) -> str:
    """
    chat_history: list of {"role": "user"/"assistant", "content": "..."}
    Converts to LangChain message format, invokes the agent, returns the
    final text response.
    """
    lc_messages = []
    for m in chat_history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    lc_messages.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": lc_messages})

    # The agent returns the full message list; the last AI message is the final answer
    final_messages = result["messages"]
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "I wasn't able to generate a response. Please try again."