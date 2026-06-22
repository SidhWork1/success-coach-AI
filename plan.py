"""
M7 — Daily plan generation.
M9 — Plan reconciliation (updates the plan automatically when new critical/high
signals appear after the plan was already built).

Takes the coach's available hours + all active alerts (from M6's signal system)
and produces a structured day plan: who gets seen today, what kind of session,
why, and who gets deferred to tomorrow with a reason.

The active plan + a running change log are persisted in Mem0 (under a fixed
"coach_plan" namespace) so they survive across coach sessions — the coach can
see what changed even if they weren't in the app when it happened.
"""

import os
import json
from datetime import date

import streamlit as st
from mem0 import MemoryClient

from signals import get_all_active_alerts


# ─────────────────────────────────────────────────────────────────────────────
# Session type mapping — signal_type → (session label, duration in minutes)
# ─────────────────────────────────────────────────────────────────────────────

SESSION_TYPE_MAP = {
    "emotional_distress":   ("Wellbeing check-in", 45),
    "dropout_risk":          ("Wellbeing check-in", 45),
    "academic_risk":         ("Academic support session", 30),
    "administrative_issue":  ("Quick admin help", 15),
}

DEFAULT_SESSION_TYPE = ("General check-in", 30)  # fallback for unrecognized signal_types

# Used to sort alerts so the most urgent get scheduled first
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
URGENCY_RANK  = {"today": 0, "within_2_days": 1, "within_week": 2}


def _session_type_for(signal_type: str) -> tuple[str, int]:
    return SESSION_TYPE_MAP.get(signal_type, DEFAULT_SESSION_TYPE)


def _sort_key(alert: dict):
    return (
        SEVERITY_RANK.get(alert.get("severity"), 9),
        URGENCY_RANK.get(alert.get("urgency"), 9),
    )


def dedupe_alerts_by_student(alerts: list[dict]) -> list[dict]:
    """
    A student may have multiple active alerts (e.g. flagged in more than one
    past session). Collapses to ONE alert per student — their single most
    severe/urgent one. Public version, reusable outside this module (e.g. the
    coach alerts view in app.py).
    """
    return _dedupe_by_student(alerts)


def _dedupe_by_student(alerts: list[dict]) -> list[dict]:
    """
    A student may have multiple active alerts (e.g. flagged in more than one
    past session). We only want to schedule them ONCE per day — using their
    single most severe/urgent alert as the reason for the session.
    """
    best_per_student: dict[str, dict] = {}

    for alert in alerts:
        sid = alert["student_id"]
        if sid not in best_per_student or _sort_key(alert) < _sort_key(best_per_student[sid]):
            best_per_student[sid] = alert

    return list(best_per_student.values())


def build_daily_plan(available_hours: float, student_ids: list[str], student_lookup: dict) -> dict:
    """
    Builds today's coaching plan.

    Args:
        available_hours: total hours the coach has available today
        student_ids: every student_id to check for alerts (full roster)
        student_lookup: dict of student_id -> roster row (for names)

    Returns:
        {
            "today": [ {student_id, name, session_type, duration_minutes, reason}, ... ],
            "deferred": [ {student_id, name, reason}, ... ],
            "total_minutes_available": int,
            "total_minutes_used": int,
        }
    """
    available_minutes = int(available_hours * 60)

    alerts = get_all_active_alerts(student_ids)
    alerts = _dedupe_by_student(alerts)
    alerts_sorted = sorted(alerts, key=_sort_key)

    today_plan = []
    deferred = []
    minutes_used = 0

    for alert in alerts_sorted:
        student_id = alert["student_id"]
        student = student_lookup.get(student_id, {})
        name = student.get("name", student_id)

        session_type, duration = _session_type_for(alert.get("signal_type"))

        if minutes_used + duration <= available_minutes:
            today_plan.append({
                "student_id": student_id,
                "name": name,
                "session_type": session_type,
                "duration_minutes": duration,
                "severity": alert.get("severity"),
                "urgency": alert.get("urgency"),
                "reason": alert.get("reason"),
                "next_action": alert.get("next_action"),
            })
            minutes_used += duration
        else:
            deferred.append({
                "student_id": student_id,
                "name": name,
                "severity": alert.get("severity"),
                "reason": (
                    f"Ran out of available time today (would need {duration} more min, "
                    f"only {available_minutes - minutes_used} min left). "
                    f"Original concern: {alert.get('reason')}"
                ),
            })

    return {
        "today": today_plan,
        "deferred": deferred,
        "total_minutes_available": available_minutes,
        "total_minutes_used": minutes_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M9 — Persisted plan storage (Mem0), so the plan + change log survive across
# coach sessions, not just st.session_state.
# ─────────────────────────────────────────────────────────────────────────────

PLAN_NAMESPACE = "coach_plan"  # fixed namespace — there's one coach, one active plan


def _get_mem0_api_key():
    try:
        if "MEM0_API_KEY" in st.secrets:
            return st.secrets["MEM0_API_KEY"]
    except Exception:
        pass
    return os.environ.get("MEM0_API_KEY")


@st.cache_resource
def _get_plan_mem0_client():
    return MemoryClient(api_key=_get_mem0_api_key())


def _today_str() -> str:
    return date.today().isoformat()


def save_active_plan(plan: dict, change_log: list[str] = None):
    """
    Persists today's plan (+ any change log entries) to Mem0.
    Overwrites any existing saved plan for today — we only ever track ONE
    active plan at a time, not a history of every version.
    """
    record = {
        "date": _today_str(),
        "plan": plan,
        "change_log": change_log or [],
    }
    client = _get_plan_mem0_client()
    # Clear out any previous plan record before saving the new one, so we
    # don't accumulate stale versions under the same namespace.
    try:
        existing = client.get_all(filters={"user_id": PLAN_NAMESPACE})
        for m in existing.get("results", []):
            client.delete(memory_id=m["id"])
    except Exception as e:
        print(f"Plan cleanup warning (non-fatal): {e}")

    client.add(
        messages=[{"role": "assistant", "content": json.dumps(record)}],
        user_id=PLAN_NAMESPACE,
        infer=False,
    )


def load_active_plan() -> dict:
    """
    Returns the saved plan record if one exists for TODAY, else None.
    A plan from a previous day is treated as stale and ignored.
    """
    try:
        client = _get_plan_mem0_client()
        result = client.get_all(filters={"user_id": PLAN_NAMESPACE})
        for m in result.get("results", []):
            try:
                record = json.loads(m.get("memory", "{}"))
                if record.get("date") == _today_str():
                    return record
            except json.JSONDecodeError:
                continue
        return None
    except Exception as e:
        print(f"Plan load error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# M9 — Reconciliation: update an existing plan when a new critical/high signal
# appears. Called right after M6 detects a new signal (in app.py's End Session
# handler), NOT on a separate trigger.
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_plan_with_new_signal(new_signal: dict, student_lookup: dict) -> dict:
    """
    Checks if a saved plan exists for today. If so, tries to fit the newly
    flagged student into it. Returns a dict describing what happened:

        {"action": "no_active_plan"}                    — nothing to update
        {"action": "added", "detail": str}                — fit in cleanly
        {"action": "bumped", "detail": str}                — added by bumping someone lower-priority
        {"action": "conflict", "detail": str}              — tie at the same severity, needs coach input
        {"action": "already_present"}                       — student already in today's plan
    """
    record = load_active_plan()
    if record is None:
        return {"action": "no_active_plan"}

    plan = record["plan"]
    change_log = record.get("change_log", [])

    student_id = new_signal["student_id"]
    student = student_lookup.get(student_id, {})
    name = student.get("name", student_id)

    # Already scheduled today? Nothing to do.
    if any(item["student_id"] == student_id for item in plan["today"]):
        return {"action": "already_present"}

    session_type, duration = _session_type_for(new_signal.get("signal_type"))
    remaining = plan["total_minutes_available"] - plan["total_minutes_used"]

    new_entry = {
        "student_id": student_id,
        "name": name,
        "session_type": session_type,
        "duration_minutes": duration,
        "severity": new_signal.get("severity"),
        "urgency": new_signal.get("urgency"),
        "reason": new_signal.get("reason"),
        "next_action": new_signal.get("next_action"),
    }

    # Case 1 — fits cleanly into remaining time
    if duration <= remaining:
        plan["today"].append(new_entry)
        plan["total_minutes_used"] += duration
        detail = f"{name} was added to today's plan ({session_type}, {duration} min) — fit into remaining time."
        change_log.append(detail)
        save_active_plan(plan, change_log)
        return {"action": "added", "detail": detail}

    # Case 2 — doesn't fit. Look for someone lower-priority already in today's plan to bump.
    today_sorted_lowest_first = sorted(plan["today"], key=_sort_key, reverse=True)
    bump_candidate = None
    for item in today_sorted_lowest_first:
        if _sort_key(item) > _sort_key(new_entry):  # candidate is strictly LOWER priority
            bump_candidate = item
            break

    if bump_candidate:
        plan["today"] = [i for i in plan["today"] if i["student_id"] != bump_candidate["student_id"]]
        plan["total_minutes_used"] -= bump_candidate["duration_minutes"]
        plan["deferred"].append({
            "student_id": bump_candidate["student_id"],
            "name": bump_candidate["name"],
            "severity": bump_candidate["severity"],
            "reason": (
                f"Bumped to tomorrow to make room for {name}, who came in with a higher-priority "
                f"({new_signal.get('severity')}) concern: {new_signal.get('reason')}"
            ),
        })
        plan["today"].append(new_entry)
        plan["total_minutes_used"] += duration
        detail = (
            f"{name} ({new_signal.get('severity')}) was added by bumping "
            f"{bump_candidate['name']} ({bump_candidate['severity']}) to tomorrow."
        )
        change_log.append(detail)
        save_active_plan(plan, change_log)
        return {"action": "bumped", "detail": detail}

    # Case 3 — no lower-priority candidate to bump. This is a genuine tie/conflict.
    # Don't decide — surface it to the coach.
    detail = (
        f"CONFLICT: {name} just came in as {new_signal.get('severity')} priority, but there's no "
        f"available slot and no lower-priority student to bump — everyone currently scheduled is "
        f"equally or more urgent. The coach needs to decide who gets seen today."
    )
    change_log.append(detail)
    save_active_plan(plan, change_log)  # save the conflict note even though the plan itself didn't change
    return {"action": "conflict", "detail": detail}