# 🚀 LinkedIn ATS & Job Matcher

A sophisticated tool designed to analyze and match resumes against LinkedIn job postings in real-time. This project leverages **Natural Language Processing (NLP)**, Web Scraping, and automated Data Pipelines to bridge the gap between job seekers and their ideal roles.

---

## 🛠️ Technology Stack
* **Language:** Python 3.10+
* **Interface:** Streamlit (Web Dashboard)
* **Database:** SQLite (Relational Data Storage)
* **NLP Models:** Spacy (NER), Sentence-Transformers (BERT-based)
* **Data Processing:** Pandas, BeautifulSoup, PDFPlumber

---

## 🧠 How the Algorithm Works
The core of this system is a **Semantic Comparison Engine** that goes beyond simple keyword matching. It understands the context and meaning behind your professional experience.

### Data Pipeline Stages:

1. **Ingestion & Scraping:**
   Job data is collected from LinkedIn using a custom scraper that maps job titles, descriptions, requirements, and locations directly into a structured SQL database.

2. **Resume Parsing:**
   PDF resumes are processed to extract raw text. The system uses **Spacy** for Named Entity Recognition (NER) to identify technical skills and professional milestones.

3. **Semantic Embedding:**
   Both the job descriptions and the resume are transformed into high-dimensional mathematical vectors using the `all-MiniLM-L6-v2` transformer model. This allows the system to recognize that "Data Scientist" and "Machine Learning Engineer" are related concepts, even if the exact words differ.

4. **Matching Score:**
   The system calculates the **Cosine Similarity** between the resume vector and the job vector. The result is a percentage-based score representing the semantic alignment of the candidate to the role.

5. **Skill Gap Analysis:**
   By comparing identified skills in the CV against the requirements in the job post, the system highlights missing keywords and technical competencies (Gap Analysis).

---

## 🏗️ Database Architecture
The project utilizes a relational SQLite database with the following structure:
* **jobs:** Core job posting data (Title, Company, Description).
* **skills:** A dictionary of technical competencies.
* **CV_job_matches:** Historically calculated scores for tracking progress.
* **skills_in_jobs:** A many-to-many relationship mapping skills to specific roles.

---

## 🚀 Getting Started
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Run the Application:** `streamlit run app.py`

---

## 👨‍💻 Developed By
**Omer Cohen** *Data Science Student (B.Sc) at The Open University of Israel.* *Specializing in Data Engineering, Automation, and applied Machine Learning.*

---
