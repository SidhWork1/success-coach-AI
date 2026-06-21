"""
Student data access — now reads from local CSV files instead of Google Sheets,
due to an ongoing Google account access issue.

Function names and signatures are kept IDENTICAL to the old Sheets-based version,
so app.py and anything else importing from here needs ZERO changes.

If/when Google Sheets access is restored, this file can be swapped back without
touching any other part of the codebase.
"""

import os
import csv

DATA_DIR = "data"

ROSTER_FILE        = os.path.join(DATA_DIR, "roster.csv")
SCORES_FILE        = os.path.join(DATA_DIR, "exam_scores.csv")
ATTENDANCE_FILE    = os.path.join(DATA_DIR, "attendance.csv")
EXAM_SCHEDULE_FILE = os.path.join(DATA_DIR, "exam_schedule.csv")


def _read_csv(filepath):
    """Reads a CSV file and returns a list of dicts (header row = keys)."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def find_one_row_by_id(data, student_id):
    """Use for files with one row per student (roster)."""
    student_id = str(student_id).strip()
    for row in data:
        if str(row.get("student_id", "")).strip() == student_id:
            return row
    return None


def find_all_rows_by_id(data, student_id):
    """Use for files with multiple rows per student (scores, attendance, exams)."""
    student_id = str(student_id).strip()
    return [
        row for row in data
        if str(row.get("student_id", "")).strip() == student_id
    ]


def verify_student_identity(student_id):
    """Roster lookup — single row per student. Source of truth for identity."""
    roster = _read_csv(ROSTER_FILE)
    return find_one_row_by_id(roster, student_id)


def get_all_students():
    """Returns the full roster — every student's basic info. Used by the coach view."""
    return _read_csv(ROSTER_FILE)


def fetch_student_data(student_id):
    """
    Fetches ALL rows belonging to this student across:
    - exam_scores (one row per subject)
    - attendance (one row per week)
    - exam_schedule (one row per upcoming exam)
    """
    scores_data        = _read_csv(SCORES_FILE)
    attendance_data     = _read_csv(ATTENDANCE_FILE)
    exam_schedule_data  = _read_csv(EXAM_SCHEDULE_FILE)

    return {
        "scores":        find_all_rows_by_id(scores_data, student_id),
        "attendance":    find_all_rows_by_id(attendance_data, student_id),
        "exam_schedule": find_all_rows_by_id(exam_schedule_data, student_id),
    }