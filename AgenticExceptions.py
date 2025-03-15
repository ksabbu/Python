import time
import pandas as pd
import numpy as np
import smtplib
from selenium import webdriver
from selenium.webdriver.common.by import By
from email.mime.text import MIMEText
from fuzzywuzzy import fuzz, process
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import pipeline


# -------------------------- 1️⃣ Extract Logs from Splunk --------------------------
def get_splunk_logs(search_url):
    """Scrape logs from Splunk using Selenium."""
    driver = webdriver.Chrome()
    driver.get(search_url)
    time.sleep(5)  # Wait for results to load
    errors = driver.find_elements(By.CLASS_NAME, "log-entry")  # Adjust selector
    error_list = [e.text for e in errors]
    driver.quit()
    return error_list


# -------------------------- 2️⃣ Check Past Resolutions (Hybrid Search) --------------------------
# Load past exceptions
df = pd.read_csv("past_exceptions.csv")  # Columns: Issue, App, DB, Env, Resolution


def hybrid_search(error_msg):
    """Search for past resolutions using Exact, Fuzzy, and Embedding-based matching."""
    # 1️⃣ Exact match
    if error_msg in df['Issue'].values:
        return df[df['Issue'] == error_msg]['Resolution'].values[0]

    # 2️⃣ Fuzzy matching
    best_match, score = process.extractOne(error_msg, df['Issue'].tolist(), scorer=fuzz.token_sort_ratio)
    if score > 80:  # Adjust threshold
        return df[df['Issue'] == best_match]['Resolution'].values[0]

    # 3️⃣ Embedding search
    vectorizer = TfidfVectorizer()
    vectors = normalize(vectorizer.fit_transform(df['Issue']).toarray())
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype(np.float32))

    query_vector = normalize(vectorizer.transform([error_msg]).toarray()).astype(np.float32)
    distances, indices = index.search(query_vector, k=1)

    if distances[0][0] < 0.2:  # Adjust threshold
        return df.iloc[indices[0][0]]['Resolution']

    return None  # No match found


# -------------------------- 3️⃣ If Not Found, Use LLM for Analysis --------------------------
qa_pipeline = pipeline("text-generation", model="path/to/local/llama")


def get_llm_resolution(error_msg):
    """Generate probable cause and solution using LLM."""
    prompt = f"Analyze this error: {error_msg}. What could be the root cause and resolution?"
    response = qa_pipeline(prompt, max_length=200)
    return response[0]['generated_text']


# -------------------------- 4️⃣ Compile Results & Send Email --------------------------
def send_email(results):
    """Send an email with resolved and unresolved issues."""
    body = "\n".join([f"{e} ➡ {res}" for e, res in results.items()])
    msg = MIMEText(body)
    msg["Subject"] = "Exception Analysis Report"
    msg["From"] = "alerts@yourcompany.com"
    msg["To"] = "team@yourcompany.com"

    with smtplib.SMTP("smtp.yourcompany.com") as server:
        server.sendmail(msg["From"], msg["To"], msg.as_string())


# -------------------------- 5️⃣ Run the Full Agentic AI Pipeline --------------------------
if __name__ == "__main__":
    # Get logs from Splunk
    error_list = get_splunk_logs("https://splunk-url.com")

    # Resolve errors using Hybrid Search + LLM
    resolved_errors = {e: (hybrid_search(e) or get_llm_resolution(e)) for e in error_list}

    # Send the final email report
    send_email(resolved_errors)

    print("✅ Exception handling completed & email sent!")
