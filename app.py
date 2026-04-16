# =============================================================================
# app.py — LinkedIn ATS Matcher (Streamlit UI)
#
# Flow when a user uploads a CV:
#   1. Extract text from PDF
#   2. Classify the CV with Gemini LLM → main_category, sub_category, level
#   3. Pull matching jobs from SQLite (filtered by main_category)
#   4. Score every job with a 4-component formula
#   5. Show the top 50 results as LinkedIn-style cards
#
# Scoring formula:
#   25% — Category match (main + sub)
#   30% — Skill overlap against gold_skills set
#   25% — Seniority level match (with overqualification penalty)
#   20% — Semantic cosine similarity (BERT embeddings)
# =============================================================================

import os
import re
import time
import sqlite3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env file for GOOGLE_API_KEY in local development
load_dotenv()

from sentence_transformers import SentenceTransformer, util
import spacy
import pdfplumber
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# =============================================================================
# PAGE CONFIG — must be the first Streamlit call
# =============================================================================
st.set_page_config(page_title="LinkedIn ATS Matcher", layout="wide")

# LinkedIn-style CSS: light grey background, white job cards, blue skill tags, blue buttons
st.markdown("""
    <style>
    .stApp { background-color: #f3f2ef; }

    /* White card with subtle shadow for each job result */
    .job-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Blue pill tags for matched skills */
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

    /* Grey pill tags for missing skills */
    .missing-tag { background-color: #f3f2ef; color: #666666; }

    /* LinkedIn-style rounded blue Apply button */
    div.stButton > button, div.stLinkButton > a {
        background-color: #0a66c2 !important;
        color: white !important;
        border-radius: 24px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        border: none !important;
        width: 100% !important;
        text-decoration: none !important;
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


# =============================================================================
# HEAVY MODEL LOADING — cached so they only load once per session
# Loading on cold start shows a progress bar to the user.
#
# SentenceTransformer (all-MiniLM-L6-v2):
#   - Encodes the CV text into a 384-dim vector
#   - Compared against pre-stored job vectors (bert_vector BLOB in DB)
#   - Cosine similarity = semantic match score
#
# spaCy (en_core_web_md):
#   - Used for NLP keyword extraction (nouns, proper nouns, adjectives)
#   - Extracts meaningful terms from both CV and job descriptions
#   - Lemmatizes tokens so "engineers" matches "engineer"
# =============================================================================
@st.cache_resource
def load_heavy_models():
    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text("🔄 Initializing AI engine... (first load may take a minute)")
    progress_bar.progress(10)

    status_text.text("🧠 Loading semantic language model (SentenceTransformer)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    progress_bar.progress(50)

    status_text.text("📝 Loading NLP vocabulary (spaCy)...")
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

model, nlp = load_heavy_models()


# =============================================================================
# GOLD SKILLS SET
# Curated list of ~300 recognizable technical and business skills.
# Used for exact/whole-word matching against CV and job descriptions.
# A "gold skill" found in both CV and job → contributes to the skills score.
# =============================================================================
gold_skills = {
    # Sales & Business
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

# Words to exclude from NLP keyword extraction — generic verbs, adjectives,
# HR buzzwords, and structural words that carry no meaningful signal.
# Without this filter, "experience", "strong", "build" would pollute skill matching.
junk_words = {
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


# =============================================================================
# CV CLASSIFICATION — Gemini LLM
# Reads the candidate's CV text and returns:
#   - main_category: which industry bucket they belong to
#   - sub_category:  their specific role type
#   - level:         seniority (intern / junior / senior / lead)
#
# This is then used to:
#   1. Filter jobs from the DB (only same main_category)
#   2. Compute the category and seniority score components
#
# Cached with @st.cache_data so re-uploading the same CV doesn't re-call the LLM.
# =============================================================================
@st.cache_data(show_spinner=False)
def resume_classification(resume_text):
    # Output schema — one classification for the whole CV
    class CvClassification(BaseModel):
        URL: str = Field(description="The original URL of the job")
        main_category: str = Field(description="The primary job category from the provided list")
        sub_category: str = Field(description="The specific sub-category")
        level: str = Field(description="The experience level (intern/junior/senior/lead)")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0  # Deterministic output — same CV always gives same result
    )

    # Hierarchical categories with keywords for CV classification
    # More detailed than the job categories — needed to handle edge cases in CVs
    my_categories = {
        "Data & BI": {
            "keywords": ["data", "bi", "crm", "sap", "ml", "analytics"],
            "sub_categories": {
                "Data Analyst": ["data analyst", "tableau", "power bi", "looker", "product analyst", "business analyst", "מנתח נתונים", "אנליסט"],
                "Data Engineer": ["data engineer", "etl", "pipeline", "airflow", "bigquery", "redshift", "spark", "מהנדס נתונים"],
                "Data Science": ["data scientist", "machine learning", "ml engineer", "nlp", "deep learning", "researcher", "מדען נתונים"],
                "AI Engineer": ["ai engineer", "genai", "generative ai", "llm", "langchain", "openai", "rag", "מהנדס בינה מלאכותית"],
                "Data Operations": ["data operations", "data ops"]
            }
        },
        "Software Engineering": {
            "keywords": ["software", "developer", "engineer", "fullstack", "backend", "frontend", "programming"],
            "sub_categories": {
                "Backend Dev": ["backend developer", "backend engineer", "server-side", "מפתח backend", "מפתח שרת"],
                "Frontend Dev": ["frontend developer", "frontend engineer", "react developer", "vue developer", "angular developer", "מפתח frontend"],
                "Fullstack Dev": ["fullstack developer", "full-stack developer", "full stack developer", "מפתח פולסטאק"],
                "Mobile Dev": ["ios developer", "android developer", "mobile developer", "flutter developer", "מפתח מובייל"]
            }
        },
        "Cyber & IT": {
            "keywords": ["cyber", "security", "it", "system", "cloud", "network", "infosec"],
            "sub_categories": {
                "Security Analyst": ["security analyst", "soc analyst", "penetration tester", "grc", "ciso", "vulnerability management", "אנליסט סייבר"],
                "DevOps": ["devops engineer", "site reliability", "sre", "kubernetes", "docker", "terraform", "jenkins", "ci/cd", "cloud engineer", "מהנדס devops"],
                "IT & System Admin": ["it manager", "help desk", "technical support", "sysadmin", "system administrator", "network engineer", "מנהל מערכות"]
            }
        },
        "Product & Design": {
            "keywords": ["product", "manager", "designer", "graphic", "design"],
            "sub_categories": {
                "Product Manager": ["product manager", "product owner", "program manager", "project manager", "pmo", "מנהל מוצר"],
                "UX/UI Designer": ["ux designer", "ui designer", "product designer", "user experience designer", "interaction designer", "מעצב ux"],
                "Graphic Designer": ["graphic designer", "motion designer", "visual designer", "brand designer", "מעצב גרפי"]
            }
        },
        "QA": {
            "keywords": ["qa", "quality assurance", "quality engineer", "test engineer", "sdet"],
            "sub_categories": {
                "QA Manual": ["qa manual", "manual tester", "manual qa", "quality assurance tester", "בודק תוכנה ידני"],
                "QA Automation": ["qa automation", "automation engineer", "sdet", "selenium", "playwright", "cypress", "test automation", "בודק תוכנה אוטומציה"]
            }
        },
        "Hardware & Mechanical": {
            "keywords": ["hardware", "board", "electrical", "vlsi", "asic", "fpga", "chip", "rf", "mechanical", "embedded"],
            "sub_categories": {
                "Hardware Engineer": ["hardware engineer", "board design", "circuit design", "analog design", "מהנדס חומרה"],
                "Mechanical Engineer": ["mechanical engineer", "mechanical design", "cad engineer", "מהנדס מכונות"],
                "Electrical Engineer": ["electrical engineer", "power electronics", "rf engineer", "signal processing", "מהנדס חשמל"],
                "Maintenance Technician": ["maintenance technician", "field technician", "service technician", "repair technician", "טכנאי תחזוקה"],
                "VLSI/Chip Design": ["vlsi engineer", "asic engineer", "fpga engineer", "rtl designer", "chip verification", "מהנדס vlsi"]
            }
        },
        "Business & Sales": {
            "keywords": ["sales", "business development", "sdr", "bdr", "account", "customer success"],
            "sub_categories": {
                "Sales / Account": ["account executive", "sales manager", "account manager", "regional sales", "מנהל מכירות", "מנהל תיק לקוח"],
                "SDR / BDR": ["sales development representative", "sdr", "business development representative", "bdr", "lead generation specialist"],
                "Customer Success": ["customer success manager", "csm", "client success manager", "מנהל הצלחת לקוח"]
            }
        },
        "Operations & Logistics": {
            "keywords": ["operations manager", "logistics manager", "supply chain manager", "procurement manager"],
            "sub_categories": {
                "Supply Chain": ["supply chain manager", "procurement manager", "logistics manager", "inventory manager", "מנהל שרשרת אספקה"],
                "Operations Manager": ["operations manager", "operations director", "operations coordinator", "מנהל תפעול"]
            }
        },
        "Retail & Customer Service": {
            "keywords": ["retail manager", "store manager", "customer service manager", "call center manager"],
            "sub_categories": {
                "Store Manager": ["store manager", "retail manager", "shop manager", "מנהל חנות", "מנהל סניף"],
                "Sales Associate": ["sales associate", "retail associate", "cashier", "קופאי", "מוכרן"],
                "Customer Service": ["customer service representative", "customer support specialist", "call center agent", "נציג שירות לקוחות"]
            }
        },
        "HR & Administration": {
            "keywords": ["hr", "human resources", "recruitment", "talent acquisition", "admin"],
            "sub_categories": {
                "HR Specialist": ["hr manager", "recruiter", "talent acquisition specialist", "hr business partner", "מגייס", "מנהל משאבי אנוש"],
                "Administrator": ["administrative assistant", "office manager", "executive assistant", "מנהלן", "עוזר אדמינסטרטיבי"]
            }
        },
        "Finance & Accounting": {
            "keywords": ["finance", "accountant", "accounting", "bookkeeper", "financial", "controller"],
            "sub_categories": {
                "Accountant": ["accountant", "bookkeeper", "financial analyst", "רואה חשבון", "חשבונאי"],
                "Finance Manager": ["finance manager", "financial manager", "financial controller", "cfo", "מנהל כספים"]
            }
        },
        "General": {
            "keywords": [
                "picker", "packer", "warehouse worker", "driver", "delivery", "courier",
                "cleaner", "security guard", "construction worker", "laborer",
                "מלקט", "מחסנאי", "עובד מחסן", "נהג", "שליח", "מחלק",
                "מנקה", "עובד ניקיון", "שומר", "אבטחה", "סייר", "עובד ייצור"
            ],
            "sub_categories": {
                "Blue Collar": [
                    "picker", "packer", "warehouse worker", "driver", "delivery", "courier",
                    "cleaner", "security guard", "construction worker", "laborer", "assembly worker",
                    "מלקט", "מחסנאי", "נהג", "שליח", "מנקה", "שומר", "עובד ייצור"
                ],
                "Other": ["other", "general", "unclassified"]
            }
        }
    }

    # CV classification prompt — analyzes the candidate's full resume text
    template = """You are an expert career classification system. Your task is to classify a candidate based on their resume/CV content.

⚠️ CRITICAL RULES:
1. Analyze job titles, experience, and skills in the resume — title is the strongest signal.
2. Match ONLY from the provided categories list.
3. If the candidate's background is blue-collar / physical labor, use "General" / "Blue Collar".
4. Never output "Other" — use "General" if truly unsure.
5. Be precise — the category must reflect the candidate's PRIMARY expertise.

CLASSIFICATION CATEGORIES:
{my_categories}

SENIORITY LEVELS: intern | junior | senior | lead
- "5+ years", "senior", "lead", "head of", "principal" → senior or lead
- "0-2 years", "student", "graduate", "entry level" → junior or intern
- Default: junior

RESUME TEXT:
{resume_text}

Return ONLY a valid JSON object:
{{
  "URL": "resume_classification",
  "main_category": "category name from list",
  "sub_category": "subcategory name from list",
  "level": "intern|junior|senior|lead"
}}

CRITICAL: Return ONLY JSON, no other text.
"""

    prompt = ChatPromptTemplate.from_template(template)
    parser = JsonOutputParser(pydantic_object=CvClassification)
    chain = prompt | llm | parser

    try:
        print("Classifying CV...")
        result = chain.invoke({
            "resume_text": resume_text,
            "my_categories": my_categories,
            "format_instructions": parser.get_format_instructions()
        })

        resume_main_category = result.get('main_category', 'Other')
        resume_sub_category = result.get('sub_category', 'Other')
        resume_level = result.get('level', 'junior')

        print(f"CV classified → {resume_main_category} / {resume_sub_category} / {resume_level}")
        return resume_level, resume_main_category, resume_sub_category

    except Exception as e:
        print(f"CV classification failed: {e}")
        # Safe fallback — will still score jobs but with generic category
        return "junior", "General", "Blue Collar"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_text(text):
    """
    Normalize extracted PDF text:
    - Collapse multiple spaces within lines
    - Remove blank lines
    This reduces noise in NLP processing and LLM input.
    """
    if not text:
        return ""
    lines = text.split('\n')
    return '\n'.join([" ".join(line.split()) for line in lines if line.strip()])


def extract_keywords(text):
    """
    Extract meaningful keywords from text using spaCy NLP.
    Keeps only nouns, proper nouns, and adjectives — these carry the most signal.
    Lemmatizes tokens (engineer ← engineers) for consistent matching.
    Filters out junk_words to remove generic HR and structural vocabulary.
    """
    doc = nlp(text.lower())
    keywords = {
        token.lemma_ for token in doc
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ']
        and token.lemma_ not in junk_words
        and len(token.lemma_) > 2  # Skip very short tokens
    }
    return keywords


def category_score(resume_category, resume_sub_category, job_category, job_sub_category):
    """
    Score how well the CV's category matches the job's category.
    Returns 0–100:
      100 — exact match on both main and sub category
       75 — same main category, different sub (still relevant but not ideal)
        0 — different main category entirely (essentially a mismatch)
    """
    if resume_category != job_category:
        return 0
    if job_sub_category != resume_sub_category:
        return 75  # Close match — same field, slightly different role
    return 100


# =============================================================================
# ATS MATCHER — Core scoring engine
# For each job in the DB, computes a 4-component match score against the CV.
#
# Components (weights must sum to 100%):
#   25% — category_score: main + sub category alignment
#   30% — skill_score: % of job's gold skills also found in the CV
#   25% — seniority_score: how well CV level matches job level
#   20% — semantic_contribution: cosine similarity of BERT embeddings * 0.20
#
# Seniority scoring:
#   Exact match         → 100
#   CV overqualified -1 → 80  (small penalty — overqualified candidates exist)
#   CV overqualified -2 → 60  (larger penalty)
#   CV underqualified -1→  50  (more penalized — hard to get hired without experience)
#   CV underqualified -2→   0  (very unlikely to be hired)
# =============================================================================
def ats_matcher(resume_text, jobs_df, resume_level, resume_category, resume_sub_category):
    resume_processed = resume_text.lower()
    resume_keywords = extract_keywords(resume_text)

    # Intersect CV keywords with gold_skills → set of skills the candidate has
    cv_gold_skills = resume_keywords & gold_skills

    # Encode the full CV text into a BERT embedding for semantic comparison
    resume_emb = model.encode(resume_processed, convert_to_tensor=True)

    results = []

    for idx, row in jobs_df.iterrows():
        job_desc = row['description_text']
        job_title = row['job_title']
        job_main_cat = row['main_category']
        job_sub_cat = row['sub_category']
        job_level = row['level']

        # --- Component 1: Category score (25%) ---
        category_score_value = category_score(
            resume_category, resume_sub_category, job_main_cat, job_sub_cat
        )

        # --- Component 4: Semantic score (20%) ---
        # Decode the stored BERT BLOB back to a numpy float32 vector
        # then compute cosine similarity with the CV embedding
        if pd.notna(row["bert_vector"]):
            try:
                job_emb = np.frombuffer(row["bert_vector"], dtype=np.float32)
                similarity = util.cos_sim(resume_emb, job_emb).item()  # Returns -1 to 1
                semantic_score = similarity * 100  # Scale to 0–100
            except Exception:
                semantic_score = 0  # Corrupted BLOB — treat as no signal
        else:
            semantic_score = 0  # No vector stored — skip semantic component

        # --- Component 2: Skills score (30%) ---
        job_keywords = extract_keywords(job_desc)
        job_gold_skills = job_keywords & gold_skills  # Skills the job requires
        matched_gold = cv_gold_skills & job_gold_skills  # Skills both CV and job have
        # % of job-required skills that the candidate possesses
        skill_score = (len(matched_gold) / len(job_gold_skills)) * 100 if job_gold_skills else 0

        # --- Component 3: Seniority score (25%) ---
        level_order = ["intern", "junior", "senior", "lead"]
        resume_idx = level_order.index(resume_level) if resume_level in level_order else 1
        job_idx = level_order.index(job_level.lower()) if job_level and job_level.lower() in level_order else 1
        level_diff = resume_idx - job_idx  # Positive = CV more experienced than job

        if level_diff == 0:
            seniority_score = 100   # Perfect seniority match
        elif level_diff == -1:
            seniority_score = 50    # CV underqualified by 1 level (e.g. junior applying to senior)
        elif level_diff < -1:
            seniority_score = 0     # CV underqualified by 2+ levels — very unlikely to be hired
        elif level_diff == 1:
            seniority_score = 80    # CV slightly overqualified — still viable
        else:
            seniority_score = 60    # CV very overqualified — may not be interested

        # --- Final combined score ---
        # Semantic contribution is cosine * 0.20 (max 20 points out of 100)
        semantic_contribution = semantic_score * 0.20

        combined_score = (
            0.25 * category_score_value +
            0.30 * skill_score +
            0.25 * seniority_score +
            semantic_contribution
        )

        results.append({
            'job_title': job_title,
            'company': row['company_name'],
            'url': row['URL'],
            'date': row['search_date'],
            'description': row['description_text'],
            'match_score': max(0, round(combined_score, 1)),   # Never show negative scores
            'score_category': round(category_score_value, 1),  # Shown in UI breakdown
            'score_skills': round(skill_score, 1),
            'score_seniority': round(seniority_score, 1),
            'score_semantic': round(semantic_score, 1),        # Raw 0–100 before weighting
            'matched_skills': list(matched_gold),
            'missing_skills': list(job_gold_skills - cv_gold_skills),
            'location': row['location'],
            'search_date': row['search_date'],
            'main_category': row['main_category']
        })

    # Return sorted by match score descending — best matches first
    return pd.DataFrame(results).sort_values('match_score', ascending=False)


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.markdown(
    "<h1 style='text-align: center; color: #0a66c2;'>LinkedIn Best Jobs for You</h1>",
    unsafe_allow_html=True
)

# PDF uploader — user drags in their CV
uploaded_file = st.file_uploader("העלה קורות חיים (PDF)", type="pdf")

if uploaded_file:
    # Extract all text from every page of the PDF
    with pdfplumber.open(uploaded_file) as pdf:
        raw_text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

    cleaned_cv = clean_text(raw_text)

    # Show the extracted CV text in a collapsible section for the user to verify
    with st.expander("📄 קורות החיים שלך:"):
        st.text_area("", cleaned_cv, height=450)

    # Classify the CV → get the category + level to use for job filtering and scoring
    conn = sqlite3.connect('linkedin_jobs.db')
    resume_level, resume_category, resume_sub_category = resume_classification(cleaned_cv)

    # Pull only jobs in the same main category as the CV for efficiency
    # If category is "Other" (unclassified), fall back to the full job list
    print(f"CV classified as: {resume_category}")
    if resume_category == "Other":
        jobs_df = pd.read_sql_query("SELECT * FROM jobs", conn)
    else:
        jobs_df = pd.read_sql_query(
            "SELECT * FROM jobs WHERE main_category = ?", conn, params=(resume_category,)
        )

    conn.close()

    if not jobs_df.empty:
        st.write("---")
        st.subheader("המשרות המתאימות ביותר עבורך:")

        # Run the scoring engine — may take a few seconds for large job sets
        with st.spinner("...מוצא לך את המשרות המתאימות ביותר בשבילך"):
            results_df = ats_matcher(cleaned_cv, jobs_df, resume_level, resume_category, resume_sub_category)

        # Cap at top 50 — showing more adds noise and slows rendering
        results_df = results_df.head(50)
        st.subheader("נמצאו {} משרות מתאימות עבורך".format(len(results_df)))

        # Render one card per job
        for _, row in results_df.iterrows():
            with st.container():
                # 4-column layout: score | job info | description | apply button
                c1, c2, c3, c4 = st.columns([1, 1, 3, 1])

                with c1:
                    # Color-coded match percentage: green >70%, orange >40%, red otherwise
                    score = row['match_score']
                    color = "green" if score > 70 else "orange" if score > 40 else "red"
                    st.markdown(
                        f"<h2 style='color:{color}; text-align:center;'>{int(score)}%</h2>",
                        unsafe_allow_html=True
                    )
                    st.caption("<p style='text-align:center;'>Match Score</p>", unsafe_allow_html=True)

                    # Score breakdown — shows contribution of each component
                    st.markdown(
                        f"<small>"
                        f"🏷️ Category: <b>{row['score_category']}%</b><br>"
                        f"🔧 Skills: <b>{row['score_skills']}%</b><br>"
                        f"📊 Seniority: <b>{row['score_seniority']}%</b><br>"
                        f"🧠 Semantic: <b>{row['score_semantic']}%</b>"
                        f"</small>",
                        unsafe_allow_html=True
                    )

                    # Show up to 5 missing skills so the user knows what gaps to address
                    if row['missing_skills']:
                        st.markdown(
                            f"<small>❌ Missing: {', '.join(row['missing_skills'][:5])}</small>",
                            unsafe_allow_html=True
                        )

                with c2:
                    # Job metadata
                    st.markdown(f"### {row['job_title']}")
                    st.markdown(f"**{row['company']}**")
                    st.markdown(f"**{row['location']}**")
                    st.markdown(f"**{row['search_date']}**")
                    st.markdown(f"**{row['main_category']}**")

                with c3:
                    st.write("")
                    desc = row['description'] or ""
                    # Show first 800 chars inline — enough context without cluttering
                    st.write(desc[:800])
                    # Rest available in an expander to keep the card compact
                    if len(desc) > 800:
                        with st.expander("🔍 קרא עוד"):
                            st.write(desc[800:])

                with c4:
                    st.write("")
                    # Direct apply link back to LinkedIn
                    st.link_button("Apply on LinkedIn", row['url'], use_container_width=True)

                st.markdown("---")
    else:
        st.error("לא נמצאו משרות בבסיס הנתונים.")

# Final CSS override — force white background over Streamlit's default
st.markdown("<style>.stApp { background-color: white; }</style>", unsafe_allow_html=True)
