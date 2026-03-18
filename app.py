import streamlit as st
import pdfplumber
import sqlite3
import pandas as pd
import spacy
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util



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
seniority_keywords = {
    'senior', 'sr', 'lead', 'leader', 'leading', 'manager', 'management', 
    'principal', 'head', 'vp', 'director', 'expert', 'staff', 'team lead', 'c++', 'cloud', 'cobol', 'backend', 'frontend', 'full stack', 'architect', 'solution architect', 'data architect', 'chief', 'officer', 'cto', 'ceo'
}
# Define gold skills (prioritized in gap analysis)
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
    'r programming', 'golang', 'bash', 'powershell', 'vba', 'html5', 'css3', 'php', 'node.js', 
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

def ats_matcher(resume_text, jobs_df):
    resume_processed = resume_text.lower()
    resume_keywords = extract_keywords(resume_text)
    cv_gold_skills = resume_keywords & gold_skills
    resume_emb = model.encode(resume_processed, convert_to_tensor=True)
    
    results = []
    for idx, row in jobs_df.iterrows():
        job_desc = row['description_text']
        job_title = row['job_title']
        
        # סמנטיקה
        # job_emb = model.encode(job_desc.lower(), convert_to_tensor=True)
       # --- תיקון חלק הסמנטיקה בתוך הלולאה ---
        if pd.notna(row["bert_vector"]):
            try:
                # טעינת הוקטור מהבייטים
                job_emb = np.frombuffer(row["bert_vector"], dtype=np.float32)
                
                # חישוב הדמיון
                similarity = util.cos_sim(resume_emb, job_emb).item()
                match_score = similarity * 100
            except Exception as e:
                # הגנה למקרה שה-blob פגום
                match_score = 0
        else:
            match_score = 0
        # סקילס
        job_keywords = extract_keywords(job_desc)
        job_gold_skills = job_keywords & gold_skills
        matched_gold = cv_gold_skills & job_gold_skills
        skill_percentage = (len(matched_gold) / len(job_gold_skills)) * 100 if job_gold_skills else 0
        
        # חישוב ציון משולב (הנוסחה שלך)
        combined_score = 40 + (0.6 * match_score + 0.4 * skill_percentage) * 0.6
        
        # קנס בכירות
        is_senior_job = any(word in job_title.lower() for word in seniority_keywords)
        is_cv_senior = any(word in resume_processed for word in seniority_keywords)
        if is_senior_job and not is_cv_senior:
            combined_score -= 35

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
            'search_date': row['search_date']

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
    jobs_df = pd.read_sql_query("SELECT job_title, company_name, description_text, URL, location, search_date, bert_vector FROM jobs WHERE description_text IS NOT NULL LIMIT 100", conn)
    conn.close()

    if not jobs_df.empty:
        st.write("---")
        st.subheader("המשרות המתאימות ביותר עבורך:")
        
        with st.spinner('מחשב התאמה סמנטית...'):
            results_df = ats_matcher(cleaned_cv, jobs_df)

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
                
                with c2:
                    st.markdown(f"### {row['job_title']}")
                    st.markdown(f"**{row['company']}**")
                    st.markdown(f"**{row['location']}**")
                    st.markdown(f"**{row['search_date']}**")
                   
                with c3:
                    st.write("") # ריווח
                    st.write(row['description'])
                
                with c4:
                    st.write("") # ריווח
                    st.link_button("Apply on LinkedIn", row['url'], use_container_width=True)
                st.markdown("---")
    else:
        st.error("לא נמצאו משרות בבסיס הנתונים.")

# עיצוב רקע
st.markdown("<style>.stApp { background-color: white; }</style>", unsafe_allow_html=True)