# LinkedIn ATS Matcher

An automated pipeline that scrapes LinkedIn jobs daily, classifies them with AI, and matches them against your CV using a custom ATS scoring engine — served through a Streamlit web app.

---

## How It Works — Full Pipeline

```
LinkedIn Guest API
       ↓
linkdin_scraper.py    ←  scrape → embed → classify → dedup → extract skills → upload
       ↓
linkedin_jobs.db      ←  SQLite database (persistent storage)
       ↓
upload_to_sheets.py   ←  push jobs + skills to Google Sheets
       ↓
app.py (Streamlit)    ←  user uploads CV → ATS scoring → ranked job cards
```

The pipeline runs automatically every night at **00:00 UTC** via GitHub Actions.

---

## Project Files

| File | Role |
|---|---|
| `linkdin_scraper.py` | Full scrape-to-classify pipeline (runs the whole show) |
| `upload_to_sheets.py` | Pushes jobs + skills tables to Google Sheets |
| `app.py` | Streamlit web app — CV upload, scoring, ranked results |
| `.github/workflows/daily_run.yml` | GitHub Actions — runs the pipeline every day at midnight |
| `linkedin_jobs.db` | SQLite database — all jobs, embeddings, classifications, skills |
| `requirements.txt` | Python dependencies |
| `gen-lang-client-*.json` | Google service account credentials (gitignored, never committed) |
| `.env` | Local environment — `GOOGLE_API_KEY` (gitignored) |

---

## Step-by-Step Pipeline (`linkdin_scraper.py`)

### Step 1 — Collect Job IDs

Hits LinkedIn's **guest search API** with the following filters:
- Location: Israel
- Experience levels: Entry, Associate, Mid-Senior (`f_E=2,1,3`)
- Posted: Last 24 hours (`f_TPR=r86400`)
- Pagination: up to 400 results (40 pages × 10 jobs each)

Extracts each job's ID from the `data-entity-urn` attribute on the card element.  
Stops early if a page returns no results.

---

### Step 2 — Fetch Full Job Details

For each job ID, calls the LinkedIn job posting guest API and scrapes:

| Field | Source in HTML |
|---|---|
| `job_title` | `<h2 class="top-card-layout__title">` |
| `company_name` | `<a class="topcard__org-name-link">` |
| `location` | First `<span class="topcard__flavor--bullet">` |
| `time_posted` | `<span class="posted-time-ago__text">` |
| `num_applicants` | `<span class="num-applicants__caption">` |
| `description_text` | `<div class="description__text--rich">` — full newline-separated text |

Uses a browser-like `User-Agent` header to reduce the chance of blocks.  
Sleeps **5–12 seconds randomly** between requests.  
On HTTP 429 (rate limit), sleeps **5 minutes** before continuing.

---

### Step 3 — Generate BERT Embeddings

Every job description is encoded into a **384-dimensional float32 vector** using `all-MiniLM-L6-v2` (SentenceTransformer).

The vector is stored as a raw bytes **BLOB** in SQLite. At query time in `app.py`, the bytes are decoded back to a numpy array and compared against the CV's embedding using cosine similarity.

This powers the **semantic score** component of the ATS formula.

---

### Step 4 — AI Classification (Gemini LLM)

Jobs are classified in batches of **50** using **Gemini 3.1 Flash Lite Preview**.

Before sending, descriptions are trimmed to their **requirements / qualifications section only** (capped at 800 chars). This reduces token usage to approximately **~83k tokens per chunk**.

Each job receives three classification outputs:

| Field | Example |
|---|---|
| `main_category` | Software Engineering |
| `sub_category` | Backend Dev |
| `level` | senior |

The LLM prompt includes:
- A full **Hebrew → category quick-reference table** (handles Israeli job market)
- Ordered **seniority detection rules** (title keywords take priority)
- Strict rules to prevent blue-collar misclassification of professional roles

After the LLM responds, a **rule-based post-processor** corrects obvious errors:
- Overrides `level` based on title keywords: director/VP/chief → lead, senior/principal/manager → senior, intern/student → intern
- Fixes professional titles wrongly tagged as Blue Collar → General / Other

Retry logic handles transient errors: **3 attempts**, backoff of 30 / 60 / 90 seconds on rate limits. A **25-second delay** runs between every chunk.

---

### Step 5 — Deduplicate the Database

After appending new jobs, the table is rebuilt keeping only the **latest row** per:
- `URL` — removes re-scraped jobs from previous days
- `description_text` — removes the same job reposted under a different URL

---

### Step 6 — Extract Skills

Each job description is scanned against a curated set of ~300 known skills (`gold_skills`) using **whole-word regex matching** (`\bskill\b` prevents partial matches).

Results stored in the `skills_in_jobs` table as `URL ↔ skill` pairs.  
Used in the app to display matched and missing skills per job.

Skill categories covered:
- Programming languages: Python, SQL, JavaScript, Java, C++, Go, Rust…
- ML / AI: PyTorch, TensorFlow, LangChain, RAG, LLM, Hugging Face…
- Databases: PostgreSQL, MongoDB, Snowflake, BigQuery, Redis…
- BI & Visualization: Tableau, Power BI, Looker, Grafana…
- Cloud & DevOps: AWS, Azure, Docker, Kubernetes, Terraform, CI/CD…
- Data Engineering: Spark, Airflow, dbt, Kafka, Databricks…
- Sales & Business: Salesforce, HubSpot, CRM, SDR, BDR, pipeline management…

---

### Step 7 — Upload to Google Sheets (`upload_to_sheets.py`)

Pushes two tabs to the `linkedin_israel` Google Spreadsheet:

| Tab | Contents |
|---|---|
| `jobs` | job title, company, location, date, category, sub-category, URL |
| `skills` | URL ↔ skill pairs |

Auth uses a Google service account:
- **Local dev**: reads the JSON key file from the project root
- **GitHub Actions**: reads from the `GCP_SERVICE_ACCOUNT_JSON` environment secret

---

## ATS Scoring Formula (`app.py`)

When a user uploads their CV, every job in the database is scored with this formula:

```
match_score = (0.25 × category_score)
            + (0.30 × skill_score)
            + (0.25 × seniority_score)
            + (0.20 × semantic_score)
```

---

### Category Score — 25%

Measures how well the CV's job category matches the job posting.

| Situation | Score |
|---|---|
| Same main category **and** same sub-category | 100 |
| Same main category, different sub-category | 75 |
| Different main category | 0 |

> Example: CV = `Software Engineering / Backend Dev`, Job = `Software Engineering / Fullstack Dev` → **75 points**

---

### Skill Score — 30%

Measures what percentage of the job's required gold skills the candidate has.

```
skill_score = matched_skills / job_required_skills × 100
```

Skills are extracted from both the CV and job description using spaCy NLP:
- Only nouns, proper nouns, and adjectives are kept
- Tokens are lemmatized so "engineers" matches "engineer"
- Generic junk words (experience, strong, excellent, build…) are filtered out
- Remaining tokens are intersected with the `gold_skills` set

> Example: Job requires 8 gold skills, CV has 6 of them → **75 points**

The UI shows up to 5 missing skills so the user knows exactly what gaps to address.

---

### Seniority Score — 25%

Measures how well the candidate's level fits the job's required level.

Levels in order: `intern (0) → junior (1) → senior (2) → lead (3)`

| Situation | diff | Score |
|---|---|---|
| Exact match | 0 | 100 |
| CV overqualified by 1 level | +1 | 80 |
| CV overqualified by 2+ levels | +2 | 60 |
| CV underqualified by 1 level | -1 | 50 |
| CV underqualified by 2+ levels | -2 | 0 |

Overqualification is penalized **less** than underqualification — an overqualified candidate can still get hired; an underqualified one typically cannot.

> Example: CV = `senior`, Job = `junior` → diff = +1 → **80 points**  
> Example: CV = `junior`, Job = `senior` → diff = -1 → **50 points**

---

### Semantic Score — 20%

Measures how conceptually similar the CV text is to the job description, even when exact keywords don't overlap.

```
similarity           = cosine_similarity(CV_vector, job_vector)   # range: -1 to 1
semantic_score       = similarity × 100                           # scaled to 0–100
semantic_contribution = semantic_score × 0.20                     # max 20 points
```

Both CV and job descriptions are encoded with `all-MiniLM-L6-v2`. Job vectors are pre-computed at scrape time and stored as BLOBs. The CV vector is computed at upload time.

> Example: A CV about "data pipelines and ETL" will score well against a job about "data infrastructure and ingestion" even if no exact skill keywords overlap.

---

### Score Display in the UI

Each job card shows the **total match %** (color-coded) and the full **score breakdown**:

```
87%    Match Score
🏷️ Category:   100%
🔧 Skills:      75%
📊 Seniority:   100%
🧠 Semantic:    65%
❌ Missing: docker, kubernetes, terraform
```

| Color | Meaning |
|---|---|
| 🟢 Green | Score ≥ 70% — strong match |
| 🟠 Orange | Score 40–69% — partial match |
| 🔴 Red | Score < 40% — weak match |

Top 50 results are shown, sorted by total match score descending.

---

## CV Classification

When the CV is uploaded, Gemini classifies it using the same taxonomy as the jobs.  
The classification drives two things:

1. **Job filtering** — only jobs with the same `main_category` are loaded from the DB (speeds up scoring and improves relevance)
2. **Score inputs** — the CV's `main_category`, `sub_category`, and `level` are used in the category and seniority score components

If classification fails or returns "Other", the app falls back to the full job list.

---

## GitHub Actions Automation

File: `.github/workflows/daily_run.yml`

Runs at **00:00 UTC every day**. Can also be triggered manually from the GitHub Actions UI.

```
00:00 UTC
    ↓
git checkout
    ↓
pip install dependencies
    ↓
python linkdin_scraper.py    ←  scrape + embed + classify + dedup + skills + sheets
    ↓
git commit linkedin_jobs.db  ←  push updated DB back to repo
```

### Required Secrets

Set in **GitHub → Settings → Secrets → Actions**:

| Secret | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini LLM classification |
| `GCP_SERVICE_ACCOUNT_JSON` | Google Sheets upload |

---

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md

# 2. Create .env with your Gemini API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Place Google service account JSON in the project root
# Download from: Google Cloud Console → IAM → Service Accounts → Keys → Add Key → JSON

# 4. Run the full pipeline
python linkdin_scraper.py

# 5. Launch the Streamlit app
streamlit run app.py
```

---

## Database Schema

### `jobs` table

| Column | Type | Description |
|---|---|---|
| `job_title` | TEXT | Job title as scraped |
| `company_name` | TEXT | Company name |
| `location` | TEXT | City or region |
| `time_posted` | TEXT | e.g. "2 days ago" |
| `num_applicants` | TEXT | e.g. "47 applicants" |
| `description_text` | TEXT | Full job description |
| `URL` | TEXT | LinkedIn job URL (unique identifier) |
| `search_date` | DATE | Date the job was scraped |
| `bert_vector` | BLOB | Float32 BERT embedding stored as raw bytes |
| `main_category` | TEXT | e.g. Software Engineering |
| `sub_category` | TEXT | e.g. Backend Dev |
| `level` | TEXT | intern / junior / senior / lead |

### `skills_in_jobs` table

| Column | Type | Description |
|---|---|---|
| `URL` | TEXT | LinkedIn job URL |
| `skill` | TEXT | Skill name found in the description |

---

## Developed By

**Omer Cohen** — Data Science Student (B.Sc), The Open University of Israel.  
Specializing in Data Engineering, Automation, and Applied Machine Learning.
