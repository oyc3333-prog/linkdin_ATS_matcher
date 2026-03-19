import sqlite3
import pandas as pd
import gspread
import os
import json
from oauth2client.service_account import ServiceAccountCredentials

# 1. חיבור לבסיס הנתונים וקריאת שתי הטבלאות
conn = sqlite3.connect('linkedin_jobs.db')

df_jobs = pd.read_sql_query("SELECT * FROM jobs", conn)
df_skills = pd.read_sql_query("SELECT * FROM skills_in_jobs", conn)

conn.close() # אפשר לסגור כבר כאן כי המידע כבר בתוך ה-DataFrames

# 2. הגדרת הרשאות גוגל
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

if os.path.exists('credentials.json'):
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
else:
    creds_dict = json.loads(os.environ['GCP_SERVICE_ACCOUNT_JSON'])
    creds = ServiceAccountCredentials.from_json_dict(creds_dict, scope)

client = gspread.authorize(creds)

# 3. פונקציה קטנה כדי לא לחזור על הקוד פעמיים
def upload_to_tab(df, tab_name):
    try:
        sheet = client.open("linkedin_israel").worksheet(tab_name)
        sheet.clear()
        
        # טיפול בערכים ריקים (NaN) כדי שלא יפילו את גוגל
        df = df.fillna("") 
        
        # הכנת הנתונים: כותרות + ערכים
        data = [df.columns.values.tolist()] + df.values.tolist()
        sheet.update(data)
        print(f"Successfully updated worksheet: {tab_name}")
    except Exception as e:
        print(f"Error in {tab_name}: {e}")

# 4. הרצה עבור שתי הלשוניות
upload_to_tab(df_jobs, "jobs")
upload_to_tab(df_skills, "skills")