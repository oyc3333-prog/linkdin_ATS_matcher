import sqlite3
import pandas as pd
import gspread
import os
import json
from google.oauth2.service_account import Credentials # <--- שינוי כאן

# 1. חיבור לבסיס הנתונים (נשאר אותו דבר)
conn = sqlite3.connect('linkedin_jobs.db')
df_jobs = pd.read_sql_query("SELECT job_title, company_name, location, search_date, main_category, sub_category, URL FROM jobs", conn)
df_skills = pd.read_sql_query("SELECT * FROM skills_in_jobs", conn)
conn.close()

# 2. הגדרת הרשאות גוגל - הדרך המעודכנת
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

CREDS_FILE = 'gen-lang-client-0461607979-8daea5a37acc.json'

if os.path.exists(CREDS_FILE):
    creds = Credentials.from_service_account_file(CREDS_FILE, scopes=scope)
elif os.path.exists('credentials.json'):
    creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
else:
    # GitHub Actions secret
    creds_info = json.loads(os.environ['GCP_SERVICE_ACCOUNT_JSON'])
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)

client = gspread.authorize(creds)

# 3. פונקציית העדכון (נשארת אותו דבר)
def upload_to_tab(df, tab_name):
    try:
        sheet = client.open("linkedin_israel").worksheet(tab_name)
        sheet.clear()
        df = df.fillna("")
        data = [df.columns.values.tolist()] + df.values.tolist()
        sheet.update(data) # בגרסאות חדשות של gspread משתמשים ב-update()
        print(f"Successfully updated: {tab_name}")
    except Exception as e:
        print(f"Error in {tab_name}: {e}")

upload_to_tab(df_jobs, "jobs")
upload_to_tab(df_skills, "skills")