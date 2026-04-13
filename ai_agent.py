import os
import time
import sqlite3
import pandas as pd

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

print("Starting AI Agent - Reclassifying All Jobs...")

# 1. הגדרת המבנה (הטופס) - הפעם עם Pydantic סטנדרטי
class JobClassification(BaseModel):
    URL: str = Field(description="The original URL of the job")
    main_category: str = Field(description="The primary job category from the provided list")
    sub_category: str = Field(description="The specific sub-category")
    level: str = Field(description="The expirement level is needed(intern/junior/senior/lead) just one value from that list!")

class jobClassClassificationList(BaseModel):
    job_list: list[JobClassification]

# כאן אתה שם את המפתח שהוצאת מ-Google AI Studio (זה שמתחיל ב-AIza)
api_key = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    api_key=api_key,
    temperature=0
)

my_categories = {
    "Data & BI": {
        "keywords": ["data", "bi", "crm", "sap", "ml", "analytics"],
        "sub_categories": {
            "Data Analyst": [ "data analyst", "tableau", "power bi", "looker", "product analyst", "business analyst"],
            "Data Engineer": ["data engineer", "etl", "pipeline", "airflow", "bigquery", "redshift", "spark"],
            "Data Science": ["scientist", "machine learning", "ml", "nlp", "deep learning", "researcher"],
            "AI Engineer": ["ai engineer", "genai", "generative ai", "llm", "langchain", "openai", "rag"],
            "Data Operations": ["operations", "ops"]
        }
    },
    "Software Engineering": {
        "keywords": ["software", "developer", "engineer", "fullstack", "backend", "frontend", "programming"],
        "sub_categories": {
            "Backend Dev": ["backend", "backend developer"],
            "Frontend Dev": ["frontend", "front", "frontend developer"],
            "Fullstack Dev": ["fullstack", "full-stack", "full stack"],
            "Mobile Dev": ["ios", "android", "mobile", "flutter"]
        }
    },
    "Cyber & IT": {
        "keywords": ["cyber", "security", "it", "system", "cloud", "network", "infosec"],
        "sub_categories": {
            "Security Analyst": ["security analyst", "soc", "penetration", "pt", "grc", "ciso", "vulnerability"],
            "DevOps": ["devops", "sre", "kubernetes", "docker", "terraform", "jenkins", "ci/cd"],
            "IT & System Admin": ["it", "help desk", "support", "sysadmin", "system administrator", "network engineer"]
        }
    },
    "Product & Design": {
        "keywords": ["product", "manager", "designer", "graphic", "design"],
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
    "Hardware & Mechanical": {
        "keywords": ["hardware", "board", "electrical", "vlsi", "asic", "fpga", "chip", "rf", "mechanical", "engineer", "technician", "maintenance"],
        "sub_categories": {
            "Hardware Engineer": ["hardware engineer", "board design", "circuit", "analog"],
            "Mechanical Engineer": ["mechanical engineer", "mechanical technician", "machine"],
            "Electrical Engineer": ["electrical engineer", "power engineer", "rf engineer"],
            "Maintenance Technician": ["maintenance technician", "maintenance", "technician", "repair"],
            "VLSI/Chip Design": ["vlsi", "asic", "fpga", "verification engineer", "rtl"]
        }
    },
    "Business & Sales": {
        "keywords": ["sales", "business development", "sdr", "bdr", "account", "success", "B2B", "sales manager"],
        "sub_categories": {
            "Sales / Account": ["account executive", "sales manager", "ae", "account manager"],
            "SDR / BDR": ["sdr", "bdr", "business development representative", "lead generation"],
            "Customer Success": ["customer success", "csm", "client success"]
        }
    },
    "Operations & Logistics": {
        "keywords": ["operations", "logistics", "warehouse", "supply chain", "inventory", "fulfillment"],
        "sub_categories": {
            "Warehouse Operations": ["warehouse", "picker", "packer", "fulfillment"],
            "Supply Chain": ["supply chain", "procurement", "logistics"],
            "Operations Manager": ["operations manager", "operations coordinator"]
        }
    },
    "Retail & Customer Service": {
        "keywords": ["retail", "store", "customer service", "sales associate", "cashier", "shop", "commercial"],
        "sub_categories": {
            "Store Manager": ["store manager", "retail manager", "shop manager"],
            "Sales Associate": ["sales associate", "retail associate", "cashier"],
            "Customer Service": ["customer service", "customer support", "call center"]
        }
    },
    "HR & Administration": {
        "keywords": ["hr", "human resources", "recruitment", "admin", "administrative"],
        "sub_categories": {
            "HR Specialist": ["hr", "recruiter", "recruitment", "talent acquisition"],
            "Administrator": ["admin", "administrative", "office administrator"]
        }
    },
    "Finance & Accounting": {
        "keywords": ["finance", "accountant", "accounting", "bookkeeper", "financial"],
        "sub_categories": {
            "Accountant": ["accountant", "bookkeeper", "financial analyst"],
            "Finance Manager": ["finance manager", "financial manager", "controller"]
        }
    }
}

# 3. הגדרת ה-Prompt לסיווג
template = """You are an expert career classification system. Your task is to classify jobs into categories based on the job title and description.

IMPORTANT RULES:
1. Read the job TITLE first - it's usually the most accurate indicator
2. Use description as secondary confirmation
3. Match ONLY from the provided categories list
4. If no clear match exists, use "Other"
5. Be precise and careful - avoid defaults

CLASSIFICATION CATEGORIES:
{my_categories}

SENIORITY LEVELS TO USE: intern, junior, senior, lead
- If not explicitly mentioned in the job posting, default to "junior"

EXAMPLES OF CORRECT CLASSIFICATIONS:
- Job: "Store Manager" → main_category: "Retail & Customer Service", sub_category: "Store Manager", level: "senior"
- Job: "Warehouse Picker" → main_category: "Operations & Logistics", sub_category: "Warehouse Operations", level: "junior"
- Job: "Mechanical Engineer" → main_category: "Hardware & Mechanical", sub_category: "Mechanical Engineer", level: "junior"
- Job: "Maintenance Technician" → main_category: "Hardware & Mechanical", sub_category: "Maintenance Technician", level: "junior"
- Job: "Sensor Engineer" → main_category: "Hardware & Mechanical", sub_category: "Electrical Engineer", level: "junior"
- Job: "Python Data Scientist" → main_category: "Data & BI", sub_category: "Data Science", level: "junior"

JOBS TO CLASSIFY:
{jobs_json}

Return ONLY a valid JSON object with this exact structure:
{{
  "job_list": [
    {{
      "URL": "exact URL from input",
      "main_category": "category name from list",
      "sub_category": "subcategory name from list",
      "level": "intern|junior|senior|lead"
    }}
  ]
}}

CRITICAL: Return ONLY JSON, no other text.
"""

prompt = ChatPromptTemplate.from_template(template)
parser = JsonOutputParser(pydantic_object=jobClassClassificationList)
chain = prompt | llm | parser

# 1. חיבור ל-SQL ומשיכת הנתונים
conn = sqlite3.connect('linkedin_jobs.db')
query = "SELECT * FROM jobs"
df = pd.read_sql(query, conn)

print(f"Found {len(df)} jobs in database. Starting reclassification...")

# 4. הלולאה המתוקנת - איסוף הנתונים
all_jobs_results = [] # רשימה במקום דיקשיונרי
chunk_size = 50
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size]
    jobs_json = chunk[['URL', 'job_title', 'description_text']].to_json(orient='records')

    try:
        print(f"Processing chunk {i//chunk_size + 1} of {len(df)//chunk_size + 1}...")
        response = chain.invoke({
            "jobs_json": jobs_json,
            "my_categories": my_categories,
            "format_instructions": parser.get_format_instructions()
        })

        # חילוץ הרשימה מתוך התגובה והוספה לרשימה הכללית
        if 'job_list' in response:
            all_jobs_results.extend(response['job_list'])

        time.sleep(10)  # Rate limiting
    except Exception as e:
        print(f"Error in chunk {i//chunk_size + 1}: {e}")
        continue

# הפיכה ל-DataFrame בסוף
results_df = pd.DataFrame(all_jobs_results)
print(f"Total jobs reclassified: {len(results_df)}")

# עדכון ה-DataFrame המקורי עם התוצאות החדשות
if not results_df.empty:
    # יצירת מילון למיפוי מהיר לפי URL
    results_dict = {row['URL']: row for _, row in results_df.iterrows()}

    # עדכון העמודות ב-DataFrame המקורי
    df['main_category'] = df['URL'].map(lambda x: results_dict.get(x, {}).get('main_category', 'Other'))
    df['sub_category'] = df['URL'].map(lambda x: results_dict.get(x, {}).get('sub_category', 'Other'))
    df['level'] = df['URL'].map(lambda x: results_dict.get(x, {}).get('level', 'junior'))

    # שמירה חזרה ל-DB
    df.to_sql('jobs', conn, if_exists='replace', index=False)
    print("✅ Database updated with new classifications!")

    # הצגת סיכום
    print("\n--- Classification Summary ---")
    print(df['main_category'].value_counts().head(10))
    print("\n--- Sample Results ---")
    print(df[['job_title', 'main_category', 'sub_category', 'level']].head(10))

else:
    print("❌ No results to update database with.")

conn.close()
print("🎉 Reclassification complete!")


# my_categories = { "Data & BI": {
#         "keywords": ["data", "bi", "crm", "sap", "ml"],
#         "sub_categories": {
#             "Data Analyst": [ "data analyst", "tableau", "power bi", "looker", "product analyst", "business analyst"],
#             "Data Engineer": ["data engineer", "etl", "pipeline", "airflow", "bigquery", "redshift", "spark"],
#             "Data Science": ["scientist", "machine learning", "ml", "nlp", "deep learning", "researcher"],
#             "AI Engineer": ["ai engineer", "genai", "generative ai", "llm", "langchain", "openai", "rag"],
#             "Data Operations": ["operations", "ops"]
#         }
#     },
#     "Software Engineering": {
#         "keywords": ["software", "developer", "engineer", "fullstack", "backend", "frontend"],
#         "sub_categories": {
#             "Backend Dev": ["backend"],
#             "Frontend Dev": ["frontend", "front"],
#             "Fullstack Dev": ["fullstack", "full-stack", "full stack"],
#             "Mobile Dev": ["ios", "android", "mobile"]
#         }
#     },
#     "Cyber & IT": {
#         "keywords": ["cyber", "security", "it", "system", "cloud", "network", "infosec"],
#         "sub_categories": {
#             "Security Analyst": ["security analyst", "soc", "penetration", "pt", "grc", "ciso", "security analyst", "vulnerability"],
#             "DevOps": ["devops", "sre", "kubernetes", "docker", "terraform", "jenkins", "ci/cd"],
#             "IT & System Admin": ["it", "help desk", "support", "sysadmin", "system administrator", "network engineer"]
#         }
#     },
#     "Product & Design": {
#         "keywords": ["product", "manager", "designer", "graphic"],
#         "sub_categories": {
#             "Product Manager": ["product manager", "product owner", "po", "pm", "inbound", "outbound", "pmo", "project manager"],
#             "UX/UI Designer": ["ux", "ui", "product designer", "user experience", "user interface"],
#             "Graphic Designer": ["graphic", "motion", "illustrator", "photoshop", "creative designer"]
#         }
#     },
#     "QA": {
#         "keywords": ["qa", "testing", "quality", "test", "verification", "validation"],
#         "sub_categories": {
#             "QA Manual": ["manual"],
#             "QA Automation": ["automation", "sdet", "selenium", "playwright", "cypress", "aut"]
#         }
#     },
#     "Hardware": {
#         "keywords": ["hardware", "board", "electrical", "vlsi", "asic", "fpga", "chip", "rf"],
#         "sub_categories": {
#             "Hardware Engineer": ["hardware engineer", "board design", "circuit", "analog"],
#             "VLSI/Chip Design": ["vlsi", "asic", "fpga", "verification engineer", "rtl"],
#             "Electrical Engineer": ["electrical engineer", "power engineer", "rf engineer"]
#         }
#     },
#     "Business & Sales": {
#         "keywords": ["sales", "business development", "sdr", "bdr", "account", "success", "B2B"],
#         "sub_categories": {
#             "Sales / Account": ["account executive", "sales manager", "ae", "account manager"],
#             "SDR / BDR": ["sdr", "bdr", "business development representative", "lead generation"],
#             "Customer Success": ["customer success", "csm", "client success"]
#         }
#     }
# }




# # 2. בניית תבנית השאלה (Prompt)
# # שים לב איך אנחנו משתמשים ב-parser מהשלב הקודם כדי להזריק את ההוראות
# prompt = ChatPromptTemplate.from_template(
#     """אתה מומחה לסיווג משרות טכנולוגיות. 
#     עליך לסווג את המשרה הבאה לפי רשימת הקטגוריות הזו בלבד:
#     {my_categories}
#     בנוסף תסווג את רמת המשרה לפי הניסון הדרוש בה. אם אין ד. הרמות האפשריות: junior, mid, senior, intern. 
#     אם לא כתוב מפורש תסווג את המשרה בjunior.

#     הוראות פורמט (חובה לעקוב אחריהן):
#     {format_instructions}

#     פרטי המשרה לעיבוד:
#     URL: {url}
#     כותרת: {title}
#     תיאור משרה: : {description}
   
#     """
# )

# # 3. חיבור הכל ביחד ל-Chain (פס ייצור)
# # הסימן | אומר: קח את הפרומפט -> שלח ל-LLM -> תרגם עם ה-Parser
# chain = prompt | llm | parser

# print("ה-Chain מוכן לעבודה!")


# # 1. חיבור ל-SQL ומשיכת הנתונים
# # שנה את ה-connection string לפי מסד הנתונים שלך (SQLite/PostgreSQL/MySQL)
# conn = sqlite3.connect('linkedin_jobs.db')
# query = "SELECT * FROM jobs"
# df = pd.read_sql(query, conn)



# # 3. הגדרת ה-Prompt לסיווג
# template = """
# You are a career expert. Categorize the following list of jobs.
# For each job ID, provide:
# 1. Category (High-level field)
# 2. Sub-category (Specific niche)
# 3. Seniority Level (intern, Junior, Mid, Senior, Lead)
# אתה מומחה לסיווג משרות טכנולוגיות. 
#    clasificate jobs categories only from that category list!
#     {my_categories}
#    clasificate seniority level only to: intern, junior, mid, lead

#     Return the results as a JSON object with a key 'job_list' containing an array of objects.
# {format_instructions}

   
   
    
# Jobs List:
# {jobs_json}

# Return ONLY a JSON object where keys are Job IDs and values are objects with keys: 
# "category", "sub_category", "level".
# """

# prompt = ChatPromptTemplate.from_template(template)
# parser = JsonOutputParser(pydantic_object=jobClassClassificationList)
# chain = prompt | llm | parser


# # 4. הלולאה המתוקנת - איסוף הנתונים
# all_jobs_results = [] # רשימה במקום דיקשיונרי
# chunk_size = 100
# for i in range(0, len(df), chunk_size):
#     chunk = df.iloc[i:i + chunk_size]
#     jobs_json = chunk[['URL', 'job_title', 'description_text']].to_json(orient='records')
    
#     try:
#         print(f"Processing chunk {i//chunk_size + 1}...")
#         response = chain.invoke({
#             "jobs_json": jobs_json, 
#             "my_categories": my_categories, 
#             "format_instructions": parser.get_format_instructions()
#         })
        
#         # חילוץ הרשימה מתוך התגובה והוספה לרשימה הכללית
#         if 'job_list' in response:
#             all_jobs_results.extend(response['job_list'])
            
#         time.sleep(20) 
#     except Exception as e:
#         print(f"Error: {e}")

# # הפיכה ל-DataFrame בסוף
# results_df = pd.DataFrame(all_jobs_results)
# print(f"Total jobs classified: {len(results_df)}")
# print(results_df)
# df['main_category'] = results_df['main_category']
# df['sub_category'] = results_df['sub_category']
# df['level'] = results_df['level']
# print(df.head(5))
# if len(df) > 2000:
#     df.to_sql('jobs', conn, if_exists='replace', index=False)
#     print("Database updated with new classifications.")
# else:
#     print("Not updating database - less than 2000 records.")

# # # 4. עיבוד בצאנקים של 50 משרות
# # chunk_size = 10
# # results_dict = {}

# # for i in range(0, 20, chunk_size):
# #     chunk = df.iloc[i:i + chunk_size]
    
# #     # הפיכת הצאנק ל-JSON (שולחים רק עמודות רלוונטיות כדי לחסוך טוקנים)
# #     jobs_json = chunk[['URL', 'job_title', 'description_text']].to_json(orient='records')
    
# #     try:
# #         print(f"Processing chunk {i//chunk_size + 1}...")
# #         response = chain.invoke({"jobs_json": jobs_json, "my_categories": my_categories, "format_instructions": parser.get_format_instructions()})
# #         print(response)
# #         results_dict.update(response)
        
# #         # השהיה קלה למניעת חריגה מה-Rate Limit של ה-Free Tier
# #         time.sleep(0.5) 
# #     except Exception as e:
# #         print(f"Error in chunk {i}: {e}")

# # 5. עדכון ה-DataFrame המקורי
# # אנחנו ממפים את התוצאות לפי ה-ID של המשרה
# # df['category'] = df['URL'].apply(lambda x: results_dict.get(str(x), {}).get('category', 'Unknown'))
# # df['sub_category'] = df['URL'].apply(lambda x: results_dict.get(str(x), {}).get('sub_category', 'Unknown'))
# # df['level'] = df['URL'].apply(lambda x: results_dict.get(str(x), {}).get('level', 'Unknown'))

# # # 6. הצגת סיכום
# # print("\n--- Dataframe Info ---")
# # print(df.info())
# # print("\n--- Dataframe Head ---")
# # print(df[['URL', 'job_title', 'category', 'sub_category', 'level']].head())














import sqlite3
import pandas as pd



import sqlite3
import pandas as pd



# 1. התחברות ל-DB (תוודא שהנתיב נכון לקובץ שלך)
conn = sqlite3.connect("linkedin_jobs.db") 

# 2. שאילתה לשליפת הטייטלים והקטגוריות
# שים לב: השתמשתי בשמות העמודות שמופיעים בשגיאות הקודמות שלך
query = """
SELECT job_title, main_category, sub_category 
FROM jobs 
LIMIT 100
"""


df = pd.read_sql_query(query, conn)
print(df.tail(200))


