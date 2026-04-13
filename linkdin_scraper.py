
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
    level: str = Field(description="The expirement level is needed(intern/junior/senior/mid) just one value from that list!")

class jobClassClassificationList(BaseModel):
    job_list: list[JobClassification]

#set llm engine
api_key = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    api_key=api_key, 
    temperature=0
)

my_categories = { "Data & BI": {
        "keywords": ["data", "bi", "crm", "sap", "ml"],
        "sub_categories": {
            "Data Analyst": [ "data analyst", "tableau", "power bi", "looker", "product analyst", "business analyst"],
            "Data Engineer": ["data engineer", "etl", "pipeline", "airflow", "bigquery", "redshift", "spark"],
            "Data Science": ["scientist", "machine learning", "ml", "nlp", "deep learning", "researcher"],
            "AI Engineer": ["ai engineer", "genai", "generative ai", "llm", "langchain", "openai", "rag"],
            "Data Operations": ["operations", "ops"]
        }
    },
    "Software Engineering": {
        "keywords": ["software", "developer", "engineer", "fullstack", "backend", "frontend"],
        "sub_categories": {
            "Backend Dev": ["backend"],
            "Frontend Dev": ["frontend", "front"],
            "Fullstack Dev": ["fullstack", "full-stack", "full stack"],
            "Mobile Dev": ["ios", "android", "mobile"]
        }
    },
    "Cyber & IT": {
        "keywords": ["cyber", "security", "it", "system", "cloud", "network", "infosec"],
        "sub_categories": {
            "Security Analyst": ["security analyst", "soc", "penetration", "pt", "grc", "ciso", "security analyst", "vulnerability"],
            "DevOps": ["devops", "sre", "kubernetes", "docker", "terraform", "jenkins", "ci/cd"],
            "IT & System Admin": ["it", "help desk", "support", "sysadmin", "system administrator", "network engineer"]
        }
    },
    "Product & Design": {
        "keywords": ["product", "manager", "designer", "graphic"],
        "sub_categories": {
            "Product Manager": ["product manager", "product owner", "po", "pm", "inbound", "outbound", "pmo", "project manager"],
            "UX/UI Designer": ["ux", "ui", "product designer", "user experience", "user interface"],
            "Graphic Designer": ["graphic", "motion", "illustrator", "photoshop", "creative designer"]
        }
    },
    "QA": {
        "keywords": ["qa", "testing", "quality", "test", "verification", "validation"],
        "sub_categories": {
            "QA Manual": ["manual"],
            "QA Automation": ["automation", "sdet", "selenium", "playwright", "cypress", "aut"]
        }
    },
    "Hardware": {
        "keywords": ["hardware", "board", "electrical", "vlsi", "asic", "fpga", "chip", "rf"],
        "sub_categories": {
            "Hardware Engineer": ["hardware engineer", "board design", "circuit", "analog"],
            "VLSI/Chip Design": ["vlsi", "asic", "fpga", "verification engineer", "rtl"],
            "Electrical Engineer": ["electrical engineer", "power engineer", "rf engineer"]
        }
    },
    "Business & Sales": {
        "keywords": ["sales", "business development", "sdr", "bdr", "account", "success", "B2B"],
        "sub_categories": {
            "Sales / Account": ["account executive", "sales manager", "ae", "account manager"],
            "SDR / BDR": ["sdr", "bdr", "business development representative", "lead generation"],
            "Customer Success": ["customer success", "csm", "client success"]
        }
    }
}

# 1. חיבור ל-SQL ומשיכת הנתונים
# שנה את ה-connection string לפי מסד הנתונים שלך (SQLite/PostgreSQL/MySQL)




# 3. הגדרת ה-Prompt לסיווג
template = """
You are a career expert. Categorize the following list of jobs.
For each job ID, provide:
1. Category (High-level field)
2. Sub-category (Specific niche)
3. Seniority Level (intern, Junior, Senior, Lead)

   clasificate jobs categories only from that category list!
    {my_categories}
   clasificate seniority level only to: intern, junior, senior, lead

    Return the results as a JSON object with a key 'job_list' containing an array of objects.
{format_instructions}   
Jobs List:
{jobs_json}


Return ONLY a JSON object where keys are Job IDs and values are objects with keys: 
"category", "sub_category", "level".
"""



prompt = ChatPromptTemplate.from_template(template)
parser = JsonOutputParser(pydantic_object=jobClassClassificationList)
chain = prompt | llm | parser


# 4. הלולאה המתוקנת - איסוף הנתונים
all_jobs_results = [] # רשימה במקום דיקשיונרי
chunk_size = 50
for i in range(0, len(jobs_df), chunk_size):
    chunk = jobs_df.iloc[i:i + chunk_size]
    jobs_json = chunk[['URL', 'job_title', 'description_text']].to_json(orient='records')
    
    try:
        print(f"Processing chunk {i//chunk_size + 1}...")
        response = chain.invoke({
            "jobs_json": jobs_json, 
            "my_categories": my_categories, 
            "format_instructions": parser.get_format_instructions()
        })
        
        # חילוץ הרשימה מתוך התגובה והוספה לרשימה הכללית
        if 'job_list' in response:
            all_jobs_results.extend(response['job_list'])
        time.sleep(10) 

    except Exception as e:
        print(f"Error: {e}")

# הפיכה ל-DataFrame בסוף
results_df = pd.DataFrame(all_jobs_results)
print(f"Total jobs classified: {len(results_df)}")
print(results_df)

jobs_df = jobs_df.merge(results_df, left_on='URL', right_on='URL', how='left')




conn = sqlite3.connect('linkedin_jobs.db')
cursor = conn.cursor()
jobs_df.to_sql('jobs', conn, if_exists='append', index=False)

# 1. יצירת טבלה זמנית עם נתונים ייחודיים בלבד לפי ה-URL
# אנחנו משתמשים ב-GROUP BY כדי לוודא שכל URL מופיע רק פעם אחת
cursor.execute("""
    CREATE TABLE jobs_backup AS 
    SELECT * FROM jobs 
    GROUP BY URL
""")

# 2. מחיקת הטבלה המקורית (המלוכלכת)
cursor.execute("DROP TABLE jobs")

# 3. שינוי שם הטבלה הזמנית לשם המקורי
cursor.execute("ALTER TABLE jobs_backup RENAME TO jobs")


print("ה-DB נוקה מכפילויות בהצלחה באמצעות SQL!")

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
skills_in_jobs_df

