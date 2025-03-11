import os
import faiss
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

# Configure logging
logging.basicConfig(level=logging.INFO)


class ExceptionTracker:
    def __init__(self, csv_path="exceptions.csv", faiss_index_path="exceptions_index",
                 metadata_path="exceptions_meta.pkl"):
        self.csv_path = csv_path
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.index = None
        self.exceptions = []
        self.vectorizer = None

        # Load or create index
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            self.load_vector_store()
        else:
            self.create_vector_store()

    def load_exceptions(self):
        """Loads exceptions from a CSV file."""
        df = pd.read_csv(self.csv_path)
        if "exception_message" not in df.columns:
            raise ValueError("CSV file must have an 'exception_message' column.")

        return df["exception_message"].astype(str).tolist()

    def create_vector_store(self):
        """Creates a FAISS index for exception similarity search."""
        self.exceptions = self.load_exceptions()
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(self.exceptions).toarray()

        # Create FAISS index
        dimension = tfidf_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(tfidf_matrix.astype(np.float32))

        # Save index and metadata
        faiss.write_index(self.index, self.faiss_index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump((self.exceptions, self.vectorizer), f)

        logging.info(f"Indexed {len(self.exceptions)} exception messages.")

    def load_vector_store(self):
        """Loads the FAISS index and vectorizer."""
        self.index = faiss.read_index(self.faiss_index_path)
        with open(self.metadata_path, "rb") as f:
            self.exceptions, self.vectorizer = pickle.load(f)

        logging.info("Vector store loaded successfully.")

    def check_exception(self, user_exception, threshold=0.2):
        """Checks if the given exception is similar to any known exceptions."""
        query_vector = self.vectorizer.transform([user_exception]).toarray().astype(np.float32)
        distances, indices = self.index.search(query_vector, 1)  # Find closest match

        closest_distance = distances[0][0]
        closest_match = self.exceptions[indices[0][0]]

        if closest_distance < threshold:
            return f"✅ This exception has been seen before:\n{closest_match}"
        else:
            return "🆕 This is a new exception."


# Example Usage
if __name__ == "__main__":
    tracker = ExceptionTracker()

    while True:
        user_exception = input("\n🔍 Enter an exception message (or type 'exit' to quit): ").strip()
        if user_exception.lower() in ["exit", "quit"]:
            break

        result = tracker.check_exception(user_exception)
        print(f"\n🔍 **Result:** {result}\n")
