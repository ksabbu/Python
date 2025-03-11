import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from transformers import BertTokenizer, BertModel
import torch
import faiss
from llama_cpp import Llama  # Using local LLaMA 2 model
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor

# Load BERT Model & Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


def create_faiss_index(embedding_dim):
    return faiss.IndexFlatL2(embedding_dim)


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


def load_exception_data(csv_path: str):
    df = pd.read_csv(csv_path)
    embeddings = [embed_text(text) for text in df["Exception/Error"]]
    embedding_dim = embeddings[0].shape[1]
    index = create_faiss_index(embedding_dim)
    index.add(torch.vstack(embeddings).numpy())
    df["embedding"] = embeddings
    return df, index


def fetch_splunk_logs_selenium(app_url):
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(app_url)

    search_box = driver.find_element(By.NAME, "search")
    search_box.send_keys("index=errors | table _time, error_message")
    search_box.submit()

    driver.implicitly_wait(10)
    logs = driver.find_elements(By.CLASS_NAME, "result-row")
    results = [{"error_message": log.text} for log in logs]

    driver.quit()
    return results


def fetch_logs_parallel(app_urls):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_splunk_logs_selenium, app_urls))
    return [item for sublist in results for item in sublist]


def match_error(error_message: str, df, index):
    error_embedding = embed_text(error_message)
    D, I = index.search(error_embedding, 1)
    if D[0][0] < 0.8:
        return df.iloc[I[0][0]].to_dict()
    return None


def analyze_with_llm(error_message: str):
    llm = Llama(model_path="llama-2-7b-chat.ggmlv3.q4_0.bin")
    prompt = f"What could be the possible cause of this error? {error_message}"
    response = llm(prompt, max_tokens=100)
    return response["choices"][0]["text"]


def update_faiss(error_message, resolution, df, index):
    new_embedding = embed_text(error_message)
    index.add(new_embedding)
    new_entry = pd.DataFrame({
        "Exception/Error": [error_message],
        "resolution": [resolution],
        "isResolved": [False],
        "comments": ["Generated by AI"]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    return df, index


def draft_email(error: str, resolution: str, is_resolved: bool, comments: str):
    email_body = f"""
    Subject: Alert: Exception Detected - {error}

    Error Details:
    {error}

    Resolution: {resolution if resolution else 'No resolution available'}
    Status: {'Resolved' if is_resolved else 'Pending'}
    Comments: {comments if comments else 'N/A'}
    """
    return email_body


def send_email(email_body: str, recipients: list):
    msg = MIMEText(email_body)
    msg['Subject'] = "Exception Alert"
    msg['From'] = "alerts@yourdomain.com"
    msg['To'] = ", ".join(recipients)

    with smtplib.SMTP('smtp.yourmailserver.com', 587) as server:
        server.starttls()
        server.login("your-email", "your-password")
        server.sendmail(msg['From'], recipients, msg.as_string())


st.title("Agentic AI Exception Handler")

exception_df, faiss_index = load_exception_data("exceptions.csv")
app_urls = ["http://app1.splunk.com", "http://app2.splunk.com"]
recipients = ["team@example.com"]

if st.button("Fetch and Analyze Logs"):
    splunk_logs = fetch_logs_parallel(app_urls)

    for log in splunk_logs:
        error_message = log['error_message']
        matched_entry = match_error(error_message, exception_df, faiss_index)

        if matched_entry:
            resolution = matched_entry['resolution']
            is_resolved = matched_entry['isResolved']
            comments = matched_entry['comments']
        else:
            resolution = analyze_with_llm(error_message)
            is_resolved = False
            comments = "Generated by AI"
            exception_df, faiss_index = update_faiss(error_message, resolution, exception_df, faiss_index)

        email_body = draft_email(error_message, resolution, is_resolved, comments)
        send_email(email_body, recipients)
        st.write(f"Email sent for error: {error_message}")

st.write("Logs analyzed and notifications sent!")
