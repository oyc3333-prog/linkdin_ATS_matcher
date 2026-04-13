import os
import time
import re
import numpy as np
import sqlite3
import pandas as pd


from sentence_transformers import SentenceTransformer, util
import spacy

import pdfplumber
import streamlit as st
from pydantic import BaseModel, Field

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# --- טעינת מודלים (עם Caching כדי למנוע איטיות) ---
# 1. הגדרות דף (תמיד בשורה הראשונה של ה-UI)
st.set_page_config(page_title="LinkedIn ATS Matcher", layout="wide")
st.markdown("""
    <style>
    /* רקע אפור בהיר כמו בלינקדין */
    .stApp {
        background-color: #f3f2ef;
    }
    /* כרטיס משרה לבן עם צל */
    .job-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* תגיות כישורים */
    .skill-tag {
        display: inline-block;
        background-color: #e7f3ff;
        color: #0a66c2;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        margin: 2px;
        font-weight: 600;
    }
    .missing-tag {
        background-color: #f3f2ef;
        color: #666666;
    }
            /* עיצוב כפתור לינקדין */
    div.stButton > button, div.stLinkButton > a {
        background-color: #0a66c2 !important;
        color: white !important;
        border-radius: 24px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        border: none !important;
        width: 100% !important;
        text-decoration: none !important; /* חשוב ללינק */
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 40px !important;
        transition: background-color 0.2s, box-shadow 0.2s !important;
    }

    div.stButton > button:hover, div.stLinkButton > a:hover {
        background-color: #004182 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
# 2. פונקציית טעינת המודלים עם מסך הטעינה
@st.cache_resource
def load_heavy_models():
    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text("🔄 מאתחל מנוע AI... (זה עשוי לקחת דקה בפעם הראשונה)")
    progress_bar.progress(10)
    
    # טעינת SentenceTransformer
    status_text.text("🧠 טוען מודל שפה סמנטי (SentenceTransformer)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    progress_bar.progress(50)
    
    # טעינת Spacy
    status_text.text("📝 טוען מילון מונחים טכנולוגי (NLP)...")
    try:
        nlp = spacy.load('en_core_web_md')
    except:
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load('en_core_web_md')
    
    progress_bar.progress(100)
    time.sleep(1)
    
    status_text.empty()
    progress_bar.empty()
    return model, nlp

# 3. קריאה לפונקציית הטעינה - זה יקרה ברגע שהדף עולה
model, nlp = load_heavy_models()

# --- מילות מפתח (מהקוד שלך) ---

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
    'pyspark', 'sas', 'matlab', 'apex', 'ruby', 'perl', 'solidity', 'linux', 'unix',

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

# Junk words to exclude
junk_words = {
  
    # מילות מבנה, גיוס וסטטוס (כולל הטיות)
    'experience', 'experiences', 'experienced', 'experiencing', 'year', 'years', 'yr', 'yrs',
    'candidate', 'candidates', 'candidacy', 'applicant', 'applicants', 'application',
    'role', 'roles', 'position', 'positions', 'job', 'jobs', 'career', 'careers',
    'company', 'companies', 'organization', 'organizations', 'org', 'firm', 'agency',
    'requirement', 'requirements', 'required', 'require', 'requires', 'requiring',
    'responsibility', 'responsibilities', 'responsible', 'task', 'tasks', 'duty', 'duties',
    'degree', 'degrees', 'bachelor', 'bachelors', 'ba', 'bsc', 'master', 'masters', 'ma', 'msc', 'phd',
    'university', 'college', 'academic', 'education', 'background', 'qualification', 'qualifications',
    'plus', 'advantage', 'advantages', 'bonus', 'preferred', 'preference', 'priority',
    'field', 'industry', 'sectors', 'domain', 'environment', 'environments', 'space',

    # שמות תואר גנריים וסופרלטיבים (הטיות ותצורות שונות)
    'strong', 'stronger', 'strongly', 'excellent', 'excellence', 'outstanding', 'great', 'greater',
    'solid', 'solidly', 'proven', 'proactive', 'passionate', 'passion', 'motivated', 'motivation',
    'independent', 'independently', 'creative', 'creativity', 'flexible', 'flexibility',
    'successful', 'successfully', 'success', 'proven', 'outstanding', 'ideal', 'perfect',
    'basic', 'basics', 'basically', 'standard', 'standards', 'standardized', 'general',
    'practical', 'practically', 'technical', 'technically', 'professional', 'professionally',
    'high', 'higher', 'highly', 'large', 'larger', 'fast', 'faster', 'quick', 'quickly',
    'hands-on', 'handson', 'detail-oriented', 'detailed', 'details', 'attention',
    'team-player', 'collaborative', 'interpersonal', 'self-starter', 'ambitious',
    'analytical', 'analysis', 'analytic', 'analytically', 'operational', 'operations',

    # פעלים תפעוליים והטיות זמן (ה"רעש" המרכזי)
    'join', 'joins', 'joined', 'joining', 'build', 'builds', 'built', 'building',
    'create', 'creates', 'created', 'creating', 'creation', 'develop', 'develops', 'developed', 'developing',
    'support', 'supports', 'supported', 'supporting', 'maintain', 'maintains', 'maintained', 'maintaining',
    'manage', 'manages', 'managed', 'managing', 'management', 'manager', 'lead', 'leads', 'led', 'leading',
    'deliver', 'delivers', 'delivered', 'delivering', 'delivery', 'help', 'helps', 'helped', 'helping',
    'assist', 'assists', 'assisted', 'assisting', 'assistance', 'provide', 'provides', 'provided', 'providing',
    'ensure', 'ensures', 'ensured', 'ensuring', 'perform', 'performs', 'performed', 'performing',
    'check', 'checks', 'checked', 'checking', 'find', 'finds', 'found', 'finding', 'findings',
    'look', 'looks', 'looked', 'looking', 'test', 'tests', 'tested', 'testing',
    'use', 'uses', 'used', 'using', 'usage', 'apply', 'applies', 'applied', 'applying',
    'implement', 'implements', 'implemented', 'implementing', 'work', 'works', 'worked', 'working',
    'collaborate', 'collaborates', 'collaborated', 'collaborating', 'collaboration',
    'coordinate', 'coordinates', 'coordinated', 'coordinating', 'drive', 'drives', 'driven', 'driving',

    # מילים קטנות, הטיות כמות והשוואה
    'less', 'more', 'most', 'least', 'much', 'many', 'very', 'extremely', 'highly',
    'hand', 'hands', 'flow', 'flows', 'scale', 'scales', 'scaling', 'scaled',
    'issue', 'issues', 'log', 'logs', 'logging', 'mindset', 'mindsets',
    'problem', 'problems', 'solve', 'solver', 'solving', 'solution', 'solutions',
    'process', 'processes', 'processed', 'processing', 'procedure', 'procedures',
    'quality', 'qualities', 'integrity', 'standard', 'standards', 'insight', 'insights',
    'key', 'keys', 'main', 'major', 'minor', 'various', 'several', 'diverse',
    'hybrid', 'remote', 'office', 'onsite', 'daily', 'weekly', 'monthly', 'recurring',
    'show', 'shows', 'showed', 'showing', 'display', 'displays', 'practical', 'practically',
    'internal', 'external', 'global', 'local', 'wide', 'broad', 'deep', 'deeply'
}



@st.cache_data(show_spinner=False)
def resume_classification(resume_text):
    class CvClassification(BaseModel):
        URL: str = Field(description="The original URL of the job")
        main_category: str = Field(description="The primary job category from the provided list")
        sub_category: str = Field(description="The specific sub-category")
        level: str = Field(description="The expirement level is needed(intern/junior/senior/lead) just one value from that list!")



    #set llm engine
    api_key = os.environ.get("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        api_key="AIzaSyAH3K3jf1jOwUX6Zl02fm6_MOKaiczldF8", 
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




    # 3. הגדרת ה-Prompt לסיווג
    template = """
    You are a career expert. Categorize the cv.
    provide:
    1. Category (High-level field)
    2. Sub-category (Specific niche)
    3. Seniority Level (intern, Junior, Senior, Lead)

    clasificate jobs categories only from that category list!
        {my_categories}
    clasificate seniority level only to: intern, junior, senior, lead

        Return the results as a JSON object fit to 
    {format_instructions}   
    .

    cvs to classify:
    {resume_text}

    """

    

    prompt = ChatPromptTemplate.from_template(template)
    parser = JsonOutputParser(pydantic_object=CvClassification)
    chain = prompt | llm | parser


    try:
        print("Classifying CV...")
        result = chain.invoke({
            "resume_text": resume_text, # הטקסט של קורות החיים שלך
            "my_categories": my_categories,
            "format_instructions": parser.get_format_instructions()
        })

        # 2. חילוץ הערכים למשתנים בודדים
        resume_main_category = result.get('main_category')
        resume_sub_category = result.get('sub_category')
        resume_level = result.get('level')

        # הדפסה לבדיקה
        print(f"--- Classification Results ---")
        print(f"Level: {resume_level}")
        print(f"Main Category: {resume_main_category}")
        print(f"Sub Category: {resume_sub_category}")

        return resume_level, resume_main_category, resume_sub_category
    
    except Exception as e:
        print(f"Error during classification: {e}")
        return "junior", "Other", "Other"  # ערכי ברירת מחדל במקרה של שגיאה

# --- פונקציות לוגיקה (מותאמות) ---

def clean_text(text):
    if not text: return ""
    lines = text.split('\n')
    return '\n'.join([" ".join(line.split()) for line in lines if line.strip()])

def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = {token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] 
                and token.lemma_ not in junk_words and len(token.lemma_) > 2}
    return keywords


    

def category_score(resume_category, resume_sub_category, job_category, job_sub_category):
    if not resume_category == job_category:
        return -200
    if not job_sub_category == resume_sub_category:
        return 0
    return 100
        
    

def ats_matcher(resume_text, jobs_df, resume_level, resume_category, resume_sub_category):
    resume_processed = resume_text.lower()
    resume_keywords = extract_keywords(resume_text)
    cv_gold_skills = resume_keywords & gold_skills
    resume_emb = model.encode(resume_processed, convert_to_tensor=True)
    
    
    results = []
    for idx, row in jobs_df.iterrows():
        job_desc = row['description_text']
        job_title = row['job_title']
        job_main_cat= row['main_category']
        job_sub_cat= row['sub_category']
        job_level = row['level']
        # חישוב ציון קטגוריה
        category_score_value = category_score(resume_category, resume_sub_category, job_main_cat, job_sub_cat)
       # --- תיקון חלק הסמנטיקה בתוך הלולאה ---
        if pd.notna(row["bert_vector"]):
            try:
                # טעינת הוקטור מהבייטים
                job_emb = np.frombuffer(row["bert_vector"], dtype=np.float32)
                
                # חישוב ציון סמנטי
                similarity = util.cos_sim(resume_emb, job_emb).item()
                semantic_score = similarity * 100
            except Exception as e:
                # הגנה למקרה שה-blob פגום
                semantic_score = 0
        else:
            semantic_score = 0
        #חישוב ציון סקילים
        job_keywords = extract_keywords(job_desc)
        job_gold_skills = job_keywords & gold_skills
        matched_gold = cv_gold_skills & job_gold_skills
        skill_score = (len(matched_gold) / len(job_gold_skills)) * 100 if job_gold_skills else 0
        
        # חישוב ציון משולב (הנוסחה שלך)
        combined_score = 30  + (0.2 * semantic_score + 0.5 * skill_score + 0.3 * category_score_value) * 0.7
        
        # קנס בכירות
        if job_level == "Senior" and resume_level in ["intern", "junior"]:
            combined_score -= 20
        elif job_level == "Lead" and resume_level in ["intern", "junior", "senior"]:
            combined_score -= 20
        else:
            combined_score += 10  
        

        results.append({
            'job_title': job_title,
            'company': row['company_name'],
            'url': row['URL'],
            'date': row['search_date'], # שימוש בתאריך החיפוש
            'description': row['description_text'],
            'match_score': max(0, round(combined_score, 1)),
            'matched_skills': list(matched_gold),
            'missing_skills': list(job_gold_skills - cv_gold_skills),
            'location': row['location'],
            'search_date': row['search_date'],
            'main_category': row['main_category']

        })      
    
    return pd.DataFrame(results).sort_values('match_score', ascending=False)

# --- ממשק המשתמש (Frontend) ---

st.markdown("<h1 style='text-align: center; color: #0a66c2;'>LinkedIn Best Jobs for You</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("העלה קורות חיים (PDF)", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        raw_text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    
    cleaned_cv = clean_text(raw_text)
    
    with st.expander("📄 קורות החיים שלך:"):
        st.text_area("", cleaned_cv, height=450)

    # טעינת משרות מה-DB
    conn = sqlite3.connect('linkedin_jobs.db')
    resume_level, resume_category, resume_sub_category = resume_classification(cleaned_cv)
    

    query = "SELECT * FROM jobs WHERE sub_category = ?"
    print(f"resume category is {resume_category}")
    # אם לא זוהתה קטגוריה (Other), נמשוך את הכל כגיבוי
    if resume_category == "Other":
        query = "SELECT * FROM jobs"
        jobs_df = pd.read_sql_query(query, conn)
    else:
        jobs_df = pd.read_sql_query(query, conn, params=(resume_sub_category,))

    conn.close()

    if not jobs_df.empty:
        st.write("---")
        st.subheader("המשרות המתאימות ביותר עבורך:")
        
        with st.spinner("...מוצא לך את המשרות המתאימות ביותר בשבילך"):
            results_df = ats_matcher(cleaned_cv, jobs_df, resume_level, resume_category, resume_sub_category)
        st.subheader("נמצאו {} משרות מתאימות עבורך".format(len(results_df)))
        for _, row in results_df.iterrows():
            # יצירת כרטיס משרה בסגנון לינקדין
            with st.container():
                c1, c2, c3, c4 = st.columns([1, 1, 3, 1])
                
                with c1:
                    # מדד התאמה (Score Circular Progress)
                    score = row['match_score']
                    color = "green" if score > 70 else "orange" if score > 40 else "red"
                    st.markdown(f"<h2 style='color:{color}; text-align:center;'>{int(score)}%</h2>", unsafe_allow_html=True)
                    st.caption("<p style='text-align:center;'>Match Score</p>", unsafe_allow_html=True)
                    if row['missing_skills']:
                        st.markdown(f"<small>Missing skills: {', '.join(row['missing_skills'])}</small>", unsafe_allow_html=True)
                        st.write(f"Total jobs in DB: {len(jobs_df)}")
                with c2:
                    st.markdown(f"### {row['job_title']}")
                    st.markdown(f"**{row['company']}**")
                    st.markdown(f"**{row['location']}**")
                    st.markdown(f"**{row['search_date']}**")
                    st.markdown(f"**{row['main_category']}**")
                   
                with c3:
                    st.write("") 
                    with st.expander("🔍 קרא את תיאור המשרה המלא"):
                        st.write(row['description'])
                
                with c4:
                    st.write("") # ריווח
                    st.link_button("Apply on LinkedIn", row['url'], use_container_width=True)
                st.markdown("---")
    else:
        st.error("לא נמצאו משרות בבסיס הנתונים.")

# עיצוב רקע
st.markdown("<style>.stApp { background-color: white; }</style>", unsafe_allow_html=True)






import sqlite3
import pandas as pd



# 1. התחברות ל-DB (תוודא שהנתיב נכון לקובץ שלך)
conn = sqlite3.connect("jobs.db") 

# 2. שאילתה לשליפת הטייטלים והקטגוריות
# שים לב: השתמשתי בשמות העמודות שמופיעים בשגיאות הקודמות שלך
query = """
SELECT job_title, main_category, sub_category 
FROM jobs 
LIMIT 100
"""


df = pd.read_sql_query(query, conn)
df.head(50)
conn.close()

