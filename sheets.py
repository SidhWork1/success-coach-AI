import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

SPREADSHEET_ID = "1vKn-9LCCcBPjfcFUgEzAWBGpsjzCJtPvrln05rR1Ht0"

ROSTER_TAB        = "roster"
SCORES_TAB        = "exam_scores"
ATTENDANCE_TAB    = "attendance"
EXAM_SCHEDULE_TAB = "exam_schedule"


def get_sheets_service():
    try:
        has_secrets = "gcp_service_account" in st.secrets
    except Exception:
        has_secrets = False

    if has_secrets:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=SCOPES
        )
    else:
        creds = service_account.Credentials.from_service_account_file(
            "credentials.json", scopes=SCOPES
        )
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


def find_one_row_by_id(data, student_id):
    student_id = str(student_id).strip()
    for row in data:
        if str(row.get("student_id", "")).strip() == student_id:
            return row
    return None


def find_all_rows_by_id(data, student_id):
    student_id = str(student_id).strip()
    return [
        row for row in data
        if str(row.get("student_id", "")).strip() == student_id
    ]


def verify_student_identity(student_id):
    service = get_sheets_service()
    roster  = get_sheet_data(service, ROSTER_TAB)
    return find_one_row_by_id(roster, student_id)


def fetch_student_data(student_id):
    service = get_sheets_service()

    scores_data       = get_sheet_data(service, SCORES_TAB)
    attendance_data    = get_sheet_data(service, ATTENDANCE_TAB)
    exam_schedule_data = get_sheet_data(service, EXAM_SCHEDULE_TAB)

    return {
        "scores":        find_all_rows_by_id(scores_data, student_id),
        "attendance":    find_all_rows_by_id(attendance_data, student_id),
        "exam_schedule": find_all_rows_by_id(exam_schedule_data, student_id),
    }