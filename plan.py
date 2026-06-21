"""
M7 — Daily plan generation.

Takes the coach's available hours + all active alerts (from M6's signal system)
and produces a structured day plan: who gets seen today, what kind of session,
why, and who gets deferred to tomorrow with a reason.
"""

from typing import TypedDict, Optional
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