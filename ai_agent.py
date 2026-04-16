import os
import time
import sqlite3
import pandas as pd
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
print("Starting AI Agent - Reclassifying All Jobs...")

class JobClassification(BaseModel):
    URL: str = Field(description="The original URL of the job")
    main_category: str = Field(description="The primary job category from the provided list")
    sub_category: str = Field(description="The specific sub-category")
    level: str = Field(description="The experience level (intern/junior/senior/lead)")

class jobClassClassificationList(BaseModel):
    job_list: list[JobClassification]

api_key = os.getenv("GOOGLE_API_KEY")
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
4. Title contains " ii", " iii", " iv" as a suffix (e.g. "Engineer II") → senior
5. Title contains "manager" and NOT preceded by "junior" or "assistant" → senior
6. Description mentions "5+ years", "7+ years", "8+ years" → senior
7. Title/description contains "intern", "internship" → intern
8. Title/description contains "student", "graduate 2025", "graduate 2026", "סטודנט", "בוגר טרי" → intern
9. Title/description contains "junior", "jr.", "entry level", "0-2 years" → junior
10. No clear seniority signal → junior (last resort only)

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

import re

def extract_requirements(description: str) -> str:
    """Extract only the requirements/qualifications section from a job description."""
    if not description or not isinstance(description, str):
        return ""

    # Normalize whitespace
    text = description.strip()

    # Common section headers that signal requirements
    req_patterns = [
        r'(?i)(requirements?|qualifications?|what we(?:\'re| are) looking for|'
        r'who you are|what you(?:\'ll| will) need|skills? (?:&|and) experience|'
        r'must.have|experience (?:&|and) skills?|job requirements?|'
        r'מה אנחנו מחפשים|דרישות|כישורים|ניסיון נדרש|תנאי קבלה)'
        r'[:\s]*\n(.*?)(?=\n(?:[A-Z][^\n]{0,60}\n)|$)'
    ]

    # Try to extract requirements sections
    for pattern in req_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            combined = "\n".join(m if isinstance(m, str) else m[-1] for m in matches)
            # Keep it under ~800 chars
            return combined[:800].strip()

    # Fallback: return first 500 chars which usually contain the role summary
    return text[:500].strip()


prompt = ChatPromptTemplate.from_template(template)
parser = JsonOutputParser(pydantic_object=jobClassClassificationList)
chain = prompt | llm | parser


def validate_classifications(df):
    """Rule-based post-processing to fix obvious LLM errors."""
    title_lower = df['job_title'].str.lower().fillna('')

    # Level overrides — order matters: lead > senior > intern, junior is default
    lead_mask = title_lower.str.contains(
        r'head of|director|\bvp\b|vice president|\bchief\b|c-level',
        regex=True
    )
    lead_mask |= title_lower.str.contains(r'\blead ', regex=True)

    senior_mask = title_lower.str.contains(
        r'\bsenior\b|\bsr\b\.|principal|\bstaff engineer\b| ii$| ii | iii | iv ',
        regex=True
    )
    manager_mask = (
        title_lower.str.contains(r'\bmanager\b', regex=True) &
        ~title_lower.str.contains(r'junior|assistant', regex=True)
    )
    senior_mask |= manager_mask

    intern_mask = title_lower.str.contains(
        r'intern|internship|student|graduate 202',
        regex=True
    )

    df.loc[lead_mask, 'level'] = 'lead'
    df.loc[senior_mask & ~lead_mask, 'level'] = 'senior'
    df.loc[intern_mask & ~lead_mask & ~senior_mask, 'level'] = 'intern'

    # Blue Collar override — professional titles should NEVER be Blue Collar
    professional_mask = title_lower.str.contains(
        r'engineer|developer|analyst|designer|scientist|architect|consultant|'
        r'specialist|officer|lawyer|controller|researcher|programmer|administrator|'
        r'coordinator|strategist|economist|broker|planner',
        regex=True
    )
    bad_blue_collar = (df['sub_category'] == 'Blue Collar') & professional_mask
    df.loc[bad_blue_collar, 'sub_category'] = 'Other'
    df.loc[bad_blue_collar, 'main_category'] = 'General'

    fixed_level = int(lead_mask.sum()) + int((senior_mask & ~lead_mask).sum()) + int((intern_mask & ~lead_mask & ~senior_mask).sum())
    fixed_cat = int(bad_blue_collar.sum())
    print(f"  Post-processing: fixed {fixed_level} level(s), {fixed_cat} Blue Collar override(s)")
    return df


def invoke_with_retry(chain, payload, max_retries=3):
    """Retry with backoff for rate limits. Abort immediately on auth errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            err = str(e)
            if "PERMISSION_DENIED" in err or "API key" in err.lower():
                print(f"  [ERROR] Auth error - check your API key: {err[:120]}")
                return None  # No point retrying
            elif "RESOURCE_EXHAUSTED" in err or "429" in err:
                wait = 30 * (attempt + 1)
                print(f"  [WAIT] Rate limit hit. Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
            else:
                print(f"  [WARN] Unexpected error: {err[:120]}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(40)  # Wait before retrying on unexpected errors
    return None

conn = sqlite3.connect('linkedin_jobs.db')
query = "SELECT * FROM jobs"
df = pd.read_sql(query, conn)

print(f"Found {len(df)} jobs in database. Starting reclassification...")

all_jobs_results = []
chunk_size = 50
total_chunks = (len(df) + chunk_size - 1) // chunk_size

# --- TEST: run first chunk only and report token usage ---
print("\n=== TEST MODE: Running first chunk to check token usage ===")
test_chunk = df.iloc[0:chunk_size]
test_chunk_proc = test_chunk.copy()
test_chunk_proc['description_text'] = test_chunk_proc['description_text'].apply(extract_requirements)
test_jobs_json = test_chunk_proc[['URL', 'job_title', 'description_text']].to_json(orient='records')

test_payload = {
    "jobs_json": test_jobs_json,
    "my_categories": my_categories,
    "format_instructions": parser.get_format_instructions()
}

# Count tokens in the rendered prompt
test_prompt_str = template.format(**{
    "jobs_json": test_jobs_json,
    "my_categories": my_categories,
})
token_count = llm.get_num_tokens(test_prompt_str)
print(f"  Estimated input tokens for 1 chunk ({chunk_size} jobs): {token_count}")

test_response = invoke_with_retry(chain, test_payload)
if test_response and 'job_list' in test_response:
    print(f"  [TEST OK] Got {len(test_response['job_list'])} classifications")
    print(f"  Sample: {test_response['job_list'][0]}")
    all_jobs_results.extend(test_response['job_list'])
else:
    print("  [TEST FAILED] No response from first chunk")

print("=== TEST COMPLETE. Continuing with remaining chunks... ===\n")
time.sleep(25)

for i in range(chunk_size, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size]
    chunk_proc = chunk.copy()
    chunk_proc['description_text'] = chunk_proc['description_text'].apply(extract_requirements)
    jobs_json = chunk_proc[['URL', 'job_title', 'description_text']].to_json(orient='records')
    chunk_num = i // chunk_size + 1

    print(f"Processing chunk {chunk_num} of {total_chunks}...")

    response = invoke_with_retry(chain, {
        "jobs_json": jobs_json,
        "my_categories": my_categories,
        "format_instructions": parser.get_format_instructions()
    })

    if response and 'job_list' in response:
        all_jobs_results.extend(response['job_list'])
        print(f"  [OK] Chunk {chunk_num} done - {len(response['job_list'])} jobs classified")
    else:
        print(f"  [SKIP] Chunk {chunk_num} skipped or failed")

    time.sleep(25)  # 25 sec delay between chunks

results_df = pd.DataFrame(all_jobs_results)
print(f"\nTotal jobs reclassified: {len(results_df)}")

if not results_df.empty:
    results_dict = {row['URL']: row for _, row in results_df.iterrows()}

    df['main_category'] = df['URL'].map(lambda x: results_dict.get(x, {}).get('main_category', 'General'))
    df['sub_category'] = df['URL'].map(lambda x: results_dict.get(x, {}).get('sub_category', 'Other'))
    df['level'] = df['URL'].map(lambda x: results_dict.get(x, {}).get('level', 'junior'))

    print("Running post-processing validation...")
    df = validate_classifications(df)

    df.to_sql('jobs', conn, if_exists='replace', index=False)
    print("Database updated with new classifications!")

    print("\n--- Classification Summary ---")
    print(df['main_category'].value_counts().head(10))
    print("\n--- Sample Results ---")
    print(df[['job_title', 'main_category', 'sub_category', 'level']].head(10))
else:
    print("No results to update database with.")

conn.close()
print("Reclassification complete!")
