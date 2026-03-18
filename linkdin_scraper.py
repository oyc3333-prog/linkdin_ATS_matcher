
#Import dependencies
import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
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

# %%
# 3. התחברות ל-DB (יוצר את הקובץ אם הוא לא קיים)
conn = sqlite3.connect('linkedin_jobs.db')

# 4. שמירת הנתונים לטבלה בשם 'jobs'
# if_exists='replace' ימחוק את הטבלה הישנה ויצור חדשה עם הנתונים מה-CSV
# אם אתה רוצה להוסיף לקיים, שנה ל-'append'
jobs_df.to_sql('jobs', conn, if_exists='append', index=False)

# 5. סגירת החיבור
conn.close()

# 1. הכנת רשימת הטקסטים (המרת העמודה לרשימה)
sentences = jobs_df['description_text'].str.lower().tolist()

# 2. יצירת הוקטורים בבת אחת (יעיל פי כמה וכמה)
# נשתמש ב-convert_to_numpy=True כי SQLite ממילא צריך Bytes/Numpy
embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)

# # 3. הכנסה ל-DataFrame - פשוט משבצים את המערך שחזר
# # כל שורה במערך הופכת לתא בעמודה
# df_jobs['embedding'] = list(embeddings)


# %%
# df_jobs['embedding'] = None
# for  row in jobs_df.iterrows():
#         job_desc = row['description_text']
#         job_emb = model.encode(job_desc.lower(), convert_to_tensor=True)
#         row['embedding'] = job_emb
       
    
