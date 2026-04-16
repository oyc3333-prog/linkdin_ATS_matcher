
#Import dependencies
import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import sqlite3
import numpy as np
import re
from sentence_transformers import SentenceTransformer

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI



model = SentenceTransformer('all-MiniLM-L6-v2')

# %%
#Create an empty list to store the job postings
id_list = []
location = 'Israel'

# %%
  
for start in range(0,400,10):

    # Construct the URL for LinkedIn job search
    list_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?location={location}&f_E=2%2C1%2C3&start={start}&f_TPR=r86400&pageNum=0"

    # Send a GET request to the URL and store the response
    response = requests.get(list_url)

    #Get the HTML, parse the response and find all list items(jobs postings)
    list_data = response.text
    list_soup = BeautifulSoup(list_data, "html.parser")
    page_jobs = list_soup.find_all("li")
    if not page_jobs:
        break
    for job in page_jobs:
        base_card_div = job.find("div", {"class": "base-card"})
        if not base_card_div:
            break
        job_id = base_card_div.get("data-entity-urn").split(":")[3]
        id_list.append(job_id)
    print(f"{len(id_list)} jobs extracted from LinkedIn")
    

    time.sleep(random.uniform(0.5,2))        
        

# %%
id_list

# %%
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Referer": "https://www.linkedin.com/jobs/search?keywords=Data%20Analyst", # שינוי Referer
    "Upgrade-Insecure-Requests": "1"
}

# Initialize an empty list to store job information
job_list = []
# Loop through the list of job IDs and get each URL
for job_id in id_list:
    # Construct the URL for each job using the job ID
    
    job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    
    # Send a GET request to the job URL and parse the reponse
    try:
        job_response = requests.get(job_url, headers=headers)
        
        if job_response.status_code != 200:
            print(f"Error {job_response.status_code} for job {job_id[0]}")
            if job_response.status_code == 429:
                print("to many requests/ 5 minutes sleep")
                time.sleep(300) # עצירה ל-5 דקות אם נחסמנו
            continue
        print(job_response.status_code)
        job_soup = BeautifulSoup(job_response.text, "html.parser")
        
        # Create a dictionary to store job details
        job_post = {}
        
        # Try to extract and store the job title
        try:
            job_post["job_title"] = job_soup.find("h2", {"class":"top-card-layout__title font-sans text-lg papabear:text-xl font-bold leading-open text-color-text mb-0 topcard__title"}).text.strip()
        except:
            job_post["job_title"] = None
            
        # Try to extract and store the company name
        try:
            job_post["company_name"] = job_soup.find("a", {"class": "topcard__org-name-link topcard__flavor--black-link"}).text.strip()
        except:
            job_post["company_name"] = None
            
        
        # אנחנו מחפשים את ה-span השני בתוך ה-row של ה-flavor
        try:
        # בדרך כלל האינדקס 0 הוא המיקום (במקרה הזה Petah Tikva)
            job_post["location"] = job_soup.find_all("span", {"class": "topcard__flavor topcard__flavor--bullet"})[0].text.strip()
        except:
            job_post["location"] = None
            
        # Try to extract and store the time posted
        try:
            job_post["time_posted"] = job_soup.find("span", {"class": "posted-time-ago__text topcard__flavor--metadata"}).text.strip()
        except:
            job_post["time_posted"] = None
            
        # Try to extract and store the number of applicants
        try:
            job_post["num_applicants"] = job_soup.find("span", {"class": "num-applicants__caption topcard__flavor--metadata topcard__flavor--bullet"}).text.strip()
        except:
            job_post["num_applicants"] = None
        try:
            job_description_div = job_soup.find('div', class_='description__text--rich')
            job_post["description_text"] = job_description_div.get_text(separator='\n', strip=True)
        except:
            job_post["description_text"] = None
            
        job_post["URL"] = job_url
            
        # Append the job details to the job_list
        job_list.append(job_post)

    except Exception as e:
        print(f"Request failed: {e}")
        time.sleep(10)
        continue

    # המתנה ארוכה יותר ומשתנה
    time.sleep(random.uniform(5, 12))
    


# %%
# Create a pandas DataFrame using the list of job dictionaries 'job_list'
jobs_df = pd.DataFrame(job_list)
jobs_df

# %%
from datetime import date

# יצירת תאריך של היום בפורמט YYYY-MM-DD
today = date.today()

# הוספת העמודה ל-DataFrame
jobs_df['search_date'] = today


# 1. יצירת הוקטורים (המרה ל-List לצורך Batching)
all_embeddings = model.encode(jobs_df['description_text'].fillna("").str.lower().tolist())

# 2. הכנסה ל-DF כרשימה של מערכים
jobs_df['embedding_temp'] = list(all_embeddings)

# 3. הפיכה ל-BLOB בעזרת apply (מוכן ל-SQLite)
jobs_df['bert_vector'] = jobs_df['embedding_temp'].apply(
    lambda x: x.astype(np.float32).tobytes() if x is not None else None
)

# 4. ניקוי עמודת העזר וטעינה ל-SQL
jobs_df.drop(columns=['embedding_temp'], inplace=True)




# הצגת התוצאה
print(jobs_df.head())

# %%
jobs_df.info()




# import re

# def get_match(text, taxonomy):
#     """פונקציית עזר לחיפוש קטגוריה ותת-קטגוריה בטקסט נתון"""
#     for category, content in taxonomy.items():
#         # בדיקת קטגוריה ראשית
#         cat_pattern = r'\b(' + '|'.join([re.escape(k) for k in content["keywords"]]) + r')\b'
#         if re.search(cat_pattern, text, re.IGNORECASE):
#             # אם נמצאה קטגוריה, נחפש תת-קטגוריה
#             for sub_cat, sub_keywords in content["sub_categories"].items():
#                 sub_pattern = r'\b(' + '|'.join([re.escape(k) for k in sub_keywords]) + r')\b'
#                 if re.search(sub_pattern, text, re.IGNORECASE):
#                     return category, sub_cat
#             # אם נמצאה קטגוריה אבל לא תת-קטגוריה ספציפית
#             return category, "General"
#     return None, None

# def classify_job_hierarchical(row):
#     title = str(row['job_title']).lower()
#     description = str(row['description_text']).lower()
    
#     # ניסיון 1: חיפוש בטייטל (הכי מדויק)
#     category, sub_category = get_match(title, job_taxonomy)
    
#     # ניסיון 2: אם לא נמצא בטייטל, מחפשים בתיאור (גיבוי)
#     if not category or not sub_category:
#         category, sub_category = get_match(description, job_taxonomy)
        
#     # אם עדיין לא נמצא כלום
#     if not category:
#         category, sub_category = "Other", "General"
        
#     return pd.Series([category, sub_category])

# # 1. יצירת חיבור לבסיס הנתונים
# conn = sqlite3.connect('linkedin_jobs.db')
# # הרצה על ה-Dataframe
# jobs_df[['main_category', 'sub_category']] = jobs_df.apply(classify_job_hierarchical, axis=1)
# jobs_df.info()

# # %%
# # 3. התחברות ל-DB (יוצר את הקובץ אם הוא לא קיים)
# conn = sqlite3.connect('linkedin_jobs.db')

# # 4. שמירת הנתונים לטבלה בשם 'jobs'
# # if_exists='replace' ימחוק את הטבלה הישנה ויצור חדשה עם הנתונים מה-CSV
# # אם אתה רוצה להוסיף לקיים, שנה ל-'append'
# jobs_df.to_sql('jobs', conn, if_exists='append', index=False)






class JobClassification(BaseModel):
    URL: str = Field(description="The original URL of the job")
    main_category: str = Field(description="The primary job category from the provided list")
    sub_category: str = Field(description="The specific sub-category")
    level: str = Field(description="The experience level (intern/junior/senior/lead)")

class jobClassClassificationList(BaseModel):
    job_list: list[JobClassification]

api_key = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)

my_categories = {
    "Data & BI": ["Data Analyst", "Data Engineer", "Data Science", "AI Engineer", "Data Operations"],
    "Software Engineering": ["Backend Dev", "Frontend Dev", "Fullstack Dev", "Mobile Dev", "Embedded Software Engineer"],
    "Cyber & IT": ["Security Analyst", "DevOps", "IT & System Admin"],
    "Product & Design": ["Product Manager", "UX/UI Designer", "Graphic Designer"],
    "QA": ["QA Manual", "QA Automation"],
    "Hardware & Mechanical": ["Hardware Engineer", "Mechanical Engineer", "Electrical Engineer", "Maintenance Technician", "VLSI/Chip Design", "Process/Industrial Engineer"],
    "Business & Sales": ["Sales / Account", "SDR / BDR", "Customer Success", "Marketing"],
    "Operations & Logistics": ["Supply Chain", "Operations Manager"],
    "Retail & Customer Service": ["Store Manager", "Sales Associate", "Customer Service"],
    "HR & Administration": ["HR Specialist", "Administrator", "Legal"],
    "Finance & Accounting": ["Accountant", "Finance Manager"],
    "General": ["Blue Collar", "Other"],
}

template = """You are an expert career classification system for an Israeli job market dataset.

⚠️ CRITICAL RULES — READ ALL BEFORE CLASSIFYING:

1. TITLE FIRST: The job title is the primary signal. Use description only to resolve ambiguity.
2. HEBREW SUPPORT: Job titles and descriptions may be in Hebrew. Classify by MEANING, not by language.
3. BLUE COLLAR ONLY for physical/manual labor: warehouse picker, driver, cleaner, security guard, construction worker → "General" / "Blue Collar". NEVER assign an office or professional role to Blue Collar.
4. FALLBACK: If a professional role doesn't fit any specific category → use "General" / "Other". Do NOT use Blue Collar as a fallback.
5. PRECISE MATCH: For professional/office/tech roles, always pick the most specific matching category.

--- HEBREW → CATEGORY QUICK REFERENCE ---
מלקט / מחסנאי / עובד מחסן          → General / Blue Collar
נהג / שליח / מחלק / בלדר           → General / Blue Collar
מנקה / עובד ניקיון                   → General / Blue Collar
שומר / אבטחה / סייר                  → General / Blue Collar
עובד ייצור / עובד הרכבה / פועל        → General / Blue Collar
קופאי/ת (simple cashier role)        → Retail & Customer Service / Sales Associate
מנהל חנות / מנהל סניף               → Retail & Customer Service / Store Manager
מנהל מחסן / מנהל לוגיסטיקה          → Operations & Logistics / Operations Manager
מגייס/ת / איש גיוס                  → HR & Administration / HR Specialist
מנהל/ת משרד / עוזר/ת אדמין          → HR & Administration / Administrator
רו"ח / חשבונאי / בוקיפר              → Finance & Accounting / Accountant
מנהל כספים / CFO                    → Finance & Accounting / Finance Manager
מפתח תוכנה / מהנדס תוכנה (backend)  → Software Engineering / Backend Dev
מפתח Full Stack / פולסטאק           → Software Engineering / Fullstack Dev
מפתח frontend / ממשק                 → Software Engineering / Frontend Dev
מנהל מוצר / Product Manager          → Product & Design / Product Manager
מעצב UX/UI                          → Product & Design / UX/UI Designer
מנתח נתונים / Data Analyst            → Data & BI / Data Analyst
מדען נתונים / Data Scientist          → Data & BI / Data Science
מהנדס נתונים / Data Engineer          → Data & BI / Data Engineer
מהנדס AI / בינה מלאכותית             → Data & BI / AI Engineer
מהנדס חומרה / Hardware Engineer       → Hardware & Mechanical / Hardware Engineer
מהנדס חשמל / Electrical Engineer      → Hardware & Mechanical / Electrical Engineer
מהנדס מכונות / Mechanical Engineer    → Hardware & Mechanical / Mechanical Engineer
מהנדס תהליך / Process Engineer        → Hardware & Mechanical / Process/Industrial Engineer
מהנדס ייצור / Manufacturing Engineer  → Hardware & Mechanical / Process/Industrial Engineer
טכנאי תחזוקה / Maintenance Tech       → Hardware & Mechanical / Maintenance Technician
DevOps / ענן / Cloud Engineer         → Cyber & IT / DevOps
אנליסט סייבר / Security Analyst        → Cyber & IT / Security Analyst
מנהל מערכות / IT Manager              → Cyber & IT / IT & System Admin
QA אוטומציה / Automation Tester        → QA / QA Automation
QA ידני / Manual Tester               → QA / QA Manual
מנהל מכירות / Sales Manager           → Business & Sales / Sales / Account
SDR / BDR / נציג פיתוח עסקי           → Business & Sales / SDR / BDR
Customer Success / הצלחת לקוח         → Business & Sales / Customer Success
שיווק / מנהל שיווק / Marketing         → Business & Sales / Marketing
עורך דין / יועץ משפטי                 → HR & Administration / Legal

--- CLASSIFICATION CATEGORIES ---
{my_categories}

--- SENIORITY LEVELS ---
intern | junior | senior | lead

Use these rules IN ORDER (first match wins):
1. Title/description contains "head of", "director", "vp ", "vice president", "chief", "c-level" → lead
2. Title contains "lead " (as a modifier, e.g. "lead engineer") → lead
3. Title/description contains "senior", "sr.", "principal", "staff engineer" → senior
4. Title contains " ii", " iii", " iv" as a suffix → senior
5. Title contains "manager" and NOT preceded by "junior" or "assistant" → senior
6. Description mentions "5+ years", "7+ years", "8+ years" → senior
7. Title/description contains "intern", "internship" → intern
8. Title/description contains "student", "graduate 2025", "graduate 2026" → intern
9. Title/description contains "junior", "jr.", "entry level", "0-2 years" → junior
10. No clear seniority signal → junior

--- JOBS TO CLASSIFY ---
{jobs_json}

Return ONLY valid JSON, no other text:
{{
  "job_list": [
    {{
      "URL": "exact URL from input",
      "main_category": "category name from list above",
      "sub_category": "subcategory name from list above",
      "level": "intern|junior|senior|lead"
    }}
  ]
}}"""


def extract_requirements(description: str) -> str:
    if not description or not isinstance(description, str):
        return ""
    text = description.strip()
    req_patterns = [
        r'(?i)(requirements?|qualifications?|what we(?:\'re| are) looking for|'
        r'who you are|what you(?:\'ll| will) need|skills? (?:&|and) experience|'
        r'must.have|experience (?:&|and) skills?|job requirements?|'
        r'מה אנחנו מחפשים|דרישות|כישורים|ניסיון נדרש|תנאי קבלה)'
        r'[:\s]*\n(.*?)(?=\n(?:[A-Z][^\n]{0,60}\n)|$)'
    ]
    for pattern in req_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            combined = "\n".join(m if isinstance(m, str) else m[-1] for m in matches)
            return combined[:800].strip()
    return text[:500].strip()


def invoke_with_retry(chain, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            err = str(e)
            if "PERMISSION_DENIED" in err or "API key" in err.lower():
                print(f"  [ERROR] Auth error: {err[:120]}")
                return None
            elif "RESOURCE_EXHAUSTED" in err or "429" in err:
                wait = 30 * (attempt + 1)
                print(f"  [WAIT] Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [WARN] {err[:120]}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(40)
    return None


def validate_classifications(df):
    title_lower = df['job_title'].str.lower().fillna('')
    lead_mask = title_lower.str.contains(r'head of|director|\bvp\b|vice president|\bchief\b|c-level', regex=True)
    lead_mask |= title_lower.str.contains(r'\blead ', regex=True)
    senior_mask = title_lower.str.contains(r'\bsenior\b|\bsr\b\.|principal|\bstaff engineer\b| ii$| ii | iii | iv ', regex=True)
    manager_mask = (title_lower.str.contains(r'\bmanager\b', regex=True) & ~title_lower.str.contains(r'junior|assistant', regex=True))
    senior_mask |= manager_mask
    intern_mask = title_lower.str.contains(r'intern|internship|student|graduate 202', regex=True)
    df.loc[lead_mask, 'level'] = 'lead'
    df.loc[senior_mask & ~lead_mask, 'level'] = 'senior'
    df.loc[intern_mask & ~lead_mask & ~senior_mask, 'level'] = 'intern'
    professional_mask = title_lower.str.contains(
        r'engineer|developer|analyst|designer|scientist|architect|consultant|'
        r'specialist|officer|lawyer|controller|researcher|programmer|administrator|'
        r'coordinator|strategist|economist|broker|planner', regex=True)
    bad_blue_collar = (df['sub_category'] == 'Blue Collar') & professional_mask
    df.loc[bad_blue_collar, 'sub_category'] = 'Other'
    df.loc[bad_blue_collar, 'main_category'] = 'General'
    fixed_level = int(lead_mask.sum()) + int((senior_mask & ~lead_mask).sum()) + int((intern_mask & ~lead_mask & ~senior_mask).sum())
    print(f"  Post-processing: fixed {fixed_level} level(s), {int(bad_blue_collar.sum())} Blue Collar override(s)")
    return df


prompt = ChatPromptTemplate.from_template(template)
parser = JsonOutputParser(pydantic_object=jobClassClassificationList)
chain = prompt | llm | parser

all_jobs_results = []
chunk_size = 50
total_chunks = (len(jobs_df) + chunk_size - 1) // chunk_size

for i in range(0, len(jobs_df), chunk_size):
    chunk = jobs_df.iloc[i:i + chunk_size].copy()
    chunk['description_text'] = chunk['description_text'].apply(extract_requirements)
    jobs_json = chunk[['URL', 'job_title', 'description_text']].to_json(orient='records')
    chunk_num = i // chunk_size + 1
    print(f"Classifying chunk {chunk_num}/{total_chunks}...")

    response = invoke_with_retry(chain, {
        "jobs_json": jobs_json,
        "my_categories": my_categories,
        "format_instructions": parser.get_format_instructions()
    })

    if response and 'job_list' in response:
        all_jobs_results.extend(response['job_list'])
        print(f"  [OK] {len(response['job_list'])} jobs classified")
    else:
        print(f"  [SKIP] Chunk {chunk_num} failed")

    time.sleep(25)

results_df = pd.DataFrame(all_jobs_results)
print(f"Total jobs classified: {len(results_df)}")

jobs_df = jobs_df.merge(results_df[['URL', 'main_category', 'sub_category', 'level']], on='URL', how='left')
jobs_df['main_category'] = jobs_df['main_category'].fillna('General')
jobs_df['sub_category'] = jobs_df['sub_category'].fillna('Other')
jobs_df['level'] = jobs_df['level'].fillna('junior')

print("Running post-processing validation...")
jobs_df = validate_classifications(jobs_df)

conn = sqlite3.connect('linkedin_jobs.db')
cursor = conn.cursor()
jobs_df.to_sql('jobs', conn, if_exists='append', index=False)

# Deduplicate by URL and description_text — keep latest entry
cursor.execute("""
    CREATE TABLE jobs_clean AS
    SELECT * FROM jobs
    WHERE rowid IN (
        SELECT MAX(rowid) FROM jobs
        GROUP BY URL
    )
    AND description_text NOT IN (
        SELECT description_text FROM jobs
        GROUP BY description_text
        HAVING COUNT(*) > 1 AND MIN(rowid) != MAX(rowid)
        INTERSECT
        SELECT description_text FROM jobs
        WHERE rowid NOT IN (
            SELECT MAX(rowid) FROM jobs GROUP BY description_text
        )
    )
""")
cursor.execute("DROP TABLE jobs")
cursor.execute("ALTER TABLE jobs_clean RENAME TO jobs")
conn.commit()
print("DB deduped by URL and description!")

print("all jobs added to DB, now extracting skills...")

gold_skills = {

    #sales
    "Cold Calling", "Lead Generation", "Outbound Prospecting", "CRM Management",
    "Salesforce", "HubSpot", "B2B SaaS", "Pipeline Management", "Quota Attainment",
    "Customer Acquisition", "Relationship Building", "Negotiation", "Closing Deals",
    "Sales Cycle", "GTM Strategy", "Market Research", "Product Demo",
    "Account Management", "Objection Handling", "Inbound Leads", "Sales Funnel",
    "Business Development", "SDR", "BDR", "Hunter Mentality", "Presentation Skills",
    "Strategic Partnership", "Value-based Selling", "Revenue Growth", "KPI Driven", "sales",
    # Programming Languages & Environments
    'python', 'sql', 'javascript', 'typescript', 'java', 'scala', 'c++', 'julia', 'rust', 
    'r programming', 'golang', 'bash', 'powershell', 'vba', 'html5', 'css', 'php', 'node.js', 
    'pyspark', 'sas', 'matlab', 'apex', 'ruby', 'perl', 'solidity', 'linux', 'unix', 'QA', 'testing', 'automation testing', 'manual testing', 'selenium', 'cypress', 'jest',

    # Data Science, ML & AI
    'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow', 'keras', 'pytorch', 'xgboost', 
    'lightgbm', 'catboost', 'statsmodels', 'scipy', 'nltk', 'spacy', 'gensim', 'transformers', 
    'huggingface', 'opencv', 'k-means', 'random forest', 'gradient boosting', 'neural networks', 
    'deep learning', 'reinforcement learning', 'computer vision', 'natural language processing', 
    'nlp', 'llm', 'langchain', 'rag', 'prompt engineering', 'explainable ai', 'xai',

    # Databases & Storage
    'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle db', 
    'sql server', 'sqlite', 'mariadb', 'dynamodb', 'snowflake', 'bigquery', 'redshift', 
    'teradata', 'neo4j', 'cosmosdb', 'influxdb', 'clickhouse', 'presto', 'trino', 'hive', 
    'hbase', 'nosql', 'rdbms', 'schema-on-read', 'schema-on-write',

    # Visualization & BI
    'tableau', 'power bi', 'looker', 'google data studio', 'qlikview', 'qliksense', 
    'matplotlib', 'seaborn', 'plotly', 'd3.js', 'grafana', 'kibana', 'superset', 'dash', 
    'streamlit', 'microstrategy', 'sap businessobjects', 'dax', 'power query',

    # Data Engineering & ETL
    'apache airflow', 'dbt', 'apache kafka', 'apache spark', 'databricks', 'talend', 
    'informatica', 'pentaho', 'stitch', 'fivetran', 'alteryx', 'apache nifi', 'hadoop', 
    'mapreduce', 'etl', 'elt', 'data pipeline', 'data warehouse', 'data lake', 'data mesh', 
    'batch processing', 'stream processing', 'airflow',

    # Cloud & DevOps
    'aws', 'azure', 'google cloud platform', 'gcp', 'amazon s3', 'ec2', 'lambda', 
    'sagemaker', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'git', 
    'github', 'gitlab', 'bitbucket', 'ci/cd', 'serverless', 'microservices',

    # Statistics & Math
    'linear regression', 'logistic regression', 'hypothesis testing', 'a/b testing', 
    'bayesian statistics', 'probability theory', 'linear algebra', 'calculus', 
    'optimization', 'time series analysis', 'forecasting', 'clustering', 
    'dimension reduction', 'pca', 'anova', 'monte carlo simulation', 'statistics',

    # Product, Business & Analysis
    'kpis', 'conversion rate', 'churn analysis', 'retention', 'ltv', 'cac', 
    'customer segmentation', 'funnel analysis', 'market basket analysis', 'cohort analysis', 
    'roi', 'seo', 'sem', 'google analytics', 'amplitude', 'mixpanel', 'segment', 
    'crm', 'salesforce', 'hubspot', 'strategic decisions', 'actionable insights',

    # Development & Architecture
    'rest api', 'graphql', 'soap', 'json', 'xml', 'web scraping', 'selenium', 
    'beautifulsoup', 'oauth', 'jwt', 'agile', 'scrum', 'kanban', 'jira', 'confluence', 
    'unit testing', 'pytest', 'integration testing', 'regex', 'excel vba', 
    'object-oriented programming', 'oop', 'functional programming', 'distributed systems',
    'multi-threading', 'parallel computing', 'microservices architecture',

    # Specialized Data Skills
    'anomaly detection', 'recommendation systems', 'graph theory', 'decision trees', 
    'support vector machines', 'svm', 'ensemble learning', 'bagging', 'boosting', 
    'data normalization', 'feature engineering', 'hyperparameter tuning', 
    'cross validation', 'overfitting', 'underfitting', 'data governance', 
    'data privacy', 'gdpr', 'cybersecurity', 'blockchain', 'data mining',
    'exploratory data analysis', 'eda', 'data quality', 'data integrity'
}


# 3. פונקציית חילוץ סקילים מתיאור המשרה
def extract_skills(description, skills_set):
    if not description:
        return []
    
    found_skills = []
    desc_lower = description.lower()
    
    for skill in skills_set:
        skill_lower = skill.lower()
        # שימוש ב-Regex לחיפוש מילה שלמה בלבד (מניעת התאמות חלקיות)
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        if re.search(pattern, desc_lower):
            found_skills.append(skill)
            
    return found_skills

# 4. מעבר על המשרות ויצירת רשימת הקשרים
extracted_data = []
for _, row in jobs_df.iterrows():
    url = row['URL']
    description = row['description_text']
    
    found = extract_skills(description, gold_skills)
    for skill in found:
        extracted_data.append({'URL': url, 'skill': skill})

# 5. שמירת התוצאות לטבלה ב-DB
skills_in_jobs_df = pd.DataFrame(extracted_data)
skills_in_jobs_df.to_sql('skills_in_jobs', conn, if_exists='append', index=False)

conn.commit()
conn.close()
print(f"Extraction complete! Found {len(skills_in_jobs_df)} skill matches.")

# Auto-upload to Google Sheets
print("Uploading to Google Sheets...")
import subprocess
subprocess.run(["python", "upload_to_sheets.py"], check=True)
print("All done! Scrape → Classify → Dedup → Sheets complete.")

