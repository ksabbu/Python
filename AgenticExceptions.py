import os
import faiss
import pickle
import logging
import numpy as np
import pandas as pd
import time
import smtplib
from selenium import webdriver
from selenium.webdriver.common.by import By
from email.mime.text import MIMEText
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

# -------------------------- 1️⃣ Exception Analyzer with Vector Store --------------------------
class ExceptionAnalyzer:
    def __init__(self, csv_path="exceptions.csv", faiss_index_path="faiss_index", metadata_path="faiss_meta.pkl", similarity_threshold=0.2):
        self.csv_path = csv_path
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.similarity_threshold = similarity_threshold

        self.index = None
        self.vectorizer = None
        self.exceptions_df = pd.DataFrame()

        self.load_exceptions()
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            self.load_vector_store()
        else:
            self.create_vector_store()
    
    def load_exceptions(self):
        """Loads past exceptions from CSV."""
        try:
            if os.path.exists(self.csv_path):
                self.exceptions_df = pd.read_csv(self.csv_path)
                logging.info(f"Loaded {len(self.exceptions_df)} past exceptions.")
            else:
                logging.warning("CSV file not found. Creating a new one.")
                self.exceptions_df = pd.DataFrame(columns=["Issue", "App", "Database", "Environment", "Resolution"])
        except Exception as e:
            logging.error(f"Error loading CSV: {e}")
    
    def create_vector_store(self):
        """Creates FAISS vector store from past exceptions."""
        try:
            if self.exceptions_df.empty:
                logging.warning("No past exceptions to index.")
                return
            
            self.vectorizer = TfidfVectorizer()
            tfidf_matrix = normalize(self.vectorizer.fit_transform(self.exceptions_df["Issue"].fillna(" ")).toarray(), norm='l2')
            
            dimension = tfidf_matrix.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(tfidf_matrix.astype(np.float32))
            
            faiss.write_index(self.index, self.faiss_index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump((self.exceptions_df, self.vectorizer), f)
            
            logging.info(f"Vector store created with {len(self.exceptions_df)} exceptions.")
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
    
    def load_vector_store(self):
        """Loads FAISS index and metadata."""
        try:
            self.index = faiss.read_index(self.faiss_index_path)
            with open(self.metadata_path, "rb") as f:
                self.exceptions_df, self.vectorizer = pickle.load(f)
            logging.info(f"Vector store loaded with {len(self.exceptions_df)} exceptions.")
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
    
    def search_exception(self, error_message):
        """Hybrid search: exact, fuzzy, then embedding-based search."""
        if self.exceptions_df.empty:
            return None
        
        # **1. Exact Match Check**
        if error_message in self.exceptions_df["Issue"].values:
            return self.exceptions_df[self.exceptions_df["Issue"] == error_message]['Resolution'].values[0]

        # **2. Fuzzy Matching**
        best_match, score = process.extractOne(error_message, self.exceptions_df["Issue"].dropna(), scorer=fuzz.ratio)
        if score > 85:
            return self.exceptions_df[self.exceptions_df["Issue"] == best_match]['Resolution'].values[0]

        # **3. Embedding-Based Search**
        query_vector = normalize(self.vectorizer.transform([error_message]).toarray(), norm='l2').astype(np.float32)
        distances, indices = self.index.search(query_vector, k=1)
        if distances[0][0] < self.similarity_threshold:
            return self.exceptions_df.iloc[indices[0][0]]['Resolution']

        return None  # No match found

# -------------------------- 2️⃣ Extract Logs from Splunk --------------------------
def get_splunk_logs(search_url):
    """Scrape logs from Splunk using Selenium."""
    driver = webdriver.Chrome()
    driver.get(search_url)
    time.sleep(5)  # Wait for results to load
    errors = driver.find_elements(By.CLASS_NAME, "log-entry")  # Adjust selector
    error_list = [e.text for e in errors]
    driver.quit()
    return error_list

# -------------------------- 3️⃣ Use LLM to Generate Resolutions for New Errors --------------------------
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
    # Load the Exception Analyzer
    analyzer = ExceptionAnalyzer()

    # Get logs from Splunk
    error_list = get_splunk_logs("https://splunk-url.com")
    
    # Resolve errors using Hybrid Search + LLM
    resolved_errors = {}
    for error in error_list:
        resolution = analyzer.search_exception(error)
        if resolution is None:
            resolution = get_llm_resolution(error)
        resolved_errors[error] = resolution
    
    # Send the final email report
    send_email(resolved_errors)
    
    print("✅ Exception handling completed & email sent!")
