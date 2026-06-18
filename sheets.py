
import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

SPREADSHEET_ID = "1vKn-9LCCcBPjfcFUgEzAWBGpsjzCJtPvrln05rR1Ht0"

# Tab names — match these exactly to your sheet
ROSTER_TAB      = "roster"
SCORES_TAB     = "exam_scores"
ATTENDANCE_TAB = "attendance"
EXAMS_TAB      = "exam_schedule"


def get_sheets_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("sheets", "v4", credentials=creds)


def get_sheet_data(service, tab_name):
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=SPREADSHEET_ID, range=tab_name)
        .execute()
    )
    rows = result.get("values", [])
    if not rows:
        return []
    headers = rows[0]
    data = []
    for row in rows[1:]:
        padded = row + [""] * (len(headers) - len(row))
        data.append(dict(zip(headers, padded)))
    return data


def find_student_by_id(data, student_id):
    """Generic row finder by student_id column."""
    student_id = str(student_id).strip()
    for row in data:
        for col in ["student_id", "Student ID", "StudentID", "Roll No", "ID"]:
            if col in row and str(row[col]).strip() == student_id:
                return row
    return None


def verify_student_identity(student_id):
    """
    Looks up the roster by student_id.
    Returns the full roster row if found, None if not.
    The roster row contains: student_id, name, program, cohort, manager_email
    This is the single source of truth for who the student is.
    """
    service = get_sheets_service()
    roster  = get_sheet_data(service, ROSTER_TAB)
    return find_student_by_id(roster, student_id)


def fetch_student_data(student_id):
    """
    Fetches scores, attendance and exam data for a given student_id.
    Returns dict with all three, or None if not found anywhere.
    """
    service = get_sheets_service()

    scores_data     = get_sheet_data(service, SCORES_TAB)
    attendance_data = get_sheet_data(service, ATTENDANCE_TAB)
    exams_data      = get_sheet_data(service, EXAMS_TAB)

    student_scores     = find_student_by_id(scores_data,     student_id)
    student_attendance = find_student_by_id(attendance_data, student_id)
    student_exams      = find_student_by_id(exams_data,      student_id)

    if not student_scores and not student_attendance and not student_exams:
        return None

    return {
        "scores":     student_scores     or {},
        "attendance": student_attendance or {},
        "exams":      student_exams      or {},
    }