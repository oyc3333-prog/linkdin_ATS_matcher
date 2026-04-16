# =============================================================================
# upload_to_sheets.py — Push DB data to Google Sheets
# Reads the jobs and skills tables from SQLite and writes them to
# two tabs in the "linkedin_israel" Google Spreadsheet.
#
# Auth priority:
#   1. Local credentials JSON file (development)
#   2. GCP_SERVICE_ACCOUNT_JSON environment variable (GitHub Actions)
# =============================================================================

import os
import json
import sqlite3
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


# =============================================================================
# STEP 1 — Read data from SQLite
# jobs tab: high-level job info + classification columns
# skills tab: URL ↔ skill pairs for skills analysis
# =============================================================================
conn = sqlite3.connect('linkedin_jobs.db')

# Pull only the columns we want to show in Sheets — skip bert_vector BLOB
df_jobs = pd.read_sql_query(
    "SELECT job_title, company_name, location, search_date, main_category, sub_category, URL FROM jobs",
    conn
)

# Full skills mapping table — each row is one skill found in one job
df_skills = pd.read_sql_query("SELECT * FROM skills_in_jobs", conn)

conn.close()


# =============================================================================
# STEP 2 — Authenticate with Google Sheets API
# Requires Sheets + Drive scopes to open, read, and write spreadsheets.
#
# Local dev: place the service account JSON file in the project root.
# CI/CD (GitHub Actions): store the full JSON as the GCP_SERVICE_ACCOUNT_JSON secret.
# =============================================================================
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# The local credentials file downloaded from Google Cloud Console
CREDS_FILE = 'gen-lang-client-0461607979-8daea5a37acc.json'

if os.path.exists(CREDS_FILE):
    # Local development — use the JSON key file directly
    creds = Credentials.from_service_account_file(CREDS_FILE, scopes=scope)
elif os.path.exists('credentials.json'):
    # Fallback local file name
    creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
else:
    # GitHub Actions — credentials are injected as an environment secret
    creds_info = json.loads(os.environ['GCP_SERVICE_ACCOUNT_JSON'])
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)

client = gspread.authorize(creds)


# =============================================================================
# STEP 3 — Upload helper
# Clears the target worksheet and rewrites it entirely with fresh data.
# fillna("") converts NaN/None to empty string — Sheets can't handle NaN.
# =============================================================================
def upload_to_tab(df, tab_name):
    """
    Clear a worksheet tab and rewrite it with the given DataFrame.
    First row = column headers, subsequent rows = data.
    """
    try:
        sheet = client.open("linkedin_israel").worksheet(tab_name)
        sheet.clear()  # Wipe existing content before rewriting

        df = df.fillna("")  # Replace NaN with empty string for Sheets compatibility
        data = [df.columns.values.tolist()] + df.values.tolist()  # Header + rows
        sheet.update(data)
        print(f"Successfully updated: {tab_name}")
    except Exception as e:
        print(f"Error updating {tab_name}: {e}")


# =============================================================================
# STEP 4 — Write both tabs
# =============================================================================
upload_to_tab(df_jobs, "jobs")      # Main jobs tab
upload_to_tab(df_skills, "skills")  # Skills mapping tab
