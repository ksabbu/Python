import os
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# 📌 Load BERT Large Uncased Whole Word Masking
MODEL_NAME = "bert-large-uncased-whole-word-masking"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()  # Set to evaluation mode

# 📌 FAISS Index Parameters
embedding_dim = 1024  # BERT Large Hidden Size
nlist = 100  # Number of clusters for FAISS IVF
faiss_index = None  # This will hold our FAISS index

# 📌 Folder with text files
DATA_FOLDER = "data_files"  # Change to your folder path

# 📌 Normalize function for embeddings
def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

# 📌 Function to read text files and split into chunks
def read_text_files(folder_path, chunk_size=300):
    text_chunks = []
    
    for filename in tqdm(os.listdir(folder_path), desc="📂 Reading Files"):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
                # Split text into chunks
                words = text.split()
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i : i + chunk_size])
                    text_chunks.append(chunk)
    
    print(f"✅ Total Chunks Created: {len(text_chunks)}")
    return text_chunks

# 📌 Function to generate BERT embeddings
def generate_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    
    embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token representation
    return normalize_embedding(embedding)

# 📌 Function to create FAISS index and store embeddings
def build_faiss_index(text_chunks):
    global faiss_index
    
    embeddings = np.array([generate_embedding(chunk) for chunk in tqdm(text_chunks, desc="🔄 Generating Embeddings")], dtype=np.float32)

    # 📌 Use FAISS IndexIVFFlat (Optimized for large datasets)
    quantizer = faiss.IndexFlatIP(embedding_dim)
    faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # 📌 Train FAISS with the embeddings
    faiss_index.train(embeddings)
    faiss_index.add(embeddings)

    # 📌 Save FAISS Index for future use
    faiss.write_index(faiss_index, "faiss_index.bin")

    print(f"✅ FAISS Index Built - Total Chunks Stored: {faiss_index.ntotal}")

# 📌 Function to load FAISS Index
def load_faiss_index():
    global faiss_index
    if os.path.exists("faiss_index.bin"):
        faiss_index = faiss.read_index("faiss_index.bin")
        print(f"✅ FAISS Index Loaded - Total Chunks: {faiss_index.ntotal}")
    else:
        print("❌ No existing FAISS index found!")

# 📌 Function to retrieve similar contexts
def search_context(query, top_k=5, threshold=0.75):
    if faiss_index is None or faiss_index.ntotal == 0:
        print("❌ FAISS index is empty! Build the index first.")
        return []

    query_embedding = normalize_embedding(generate_embedding(query)).reshape(1, -1)
    scores, indices = faiss_index.search(query_embedding, top_k)

    relevant_contexts = []
    for i, score in zip(indices[0], scores[0]):
        if score >= threshold:
            relevant_contexts.append((text_chunks[i], score))

    if not relevant_contexts:
        print("❌ No relevant results found! Try lowering the threshold.")

    return sorted(relevant_contexts, key=lambda x: x[1], reverse=True)

# 📌 MAIN EXECUTION
if __name__ == "__main__":
    # Step 1: Read Text Files
    text_chunks = read_text_files(DATA_FOLDER)

    # Step 2: Build FAISS Index
    build_faiss_index(text_chunks)

    # Step 3: Load FAISS Index (If needed)
    # load_faiss_index()

    # Step 4: Search for a query
    query = "Business rule for payment processing"
    results = search_context(query, top_k=3)

    # Display results
    for i, (context, score) in enumerate(results):
        print(f"\n🔹 Result {i+1} (Score: {score:.4f}):\n{context}")
