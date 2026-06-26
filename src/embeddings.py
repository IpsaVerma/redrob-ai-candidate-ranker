import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def create_embeddings():
    file_path = '../data/processed_candidates.csv'
    
    # 1. Safety check
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found. Did you run preprocess.py?")
        return

    print("1️⃣ Loading processed candidate data...")
    df = pd.read_csv(file_path)

    # 2. Load the AI Model
    print("2️⃣ Loading the Transformer Model (Downloading if first time)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Convert Text to Vectors
    print("3️⃣ Converting resumes into mathematical vectors. Please wait...")
    # We grab the massive text block we made in preprocess.py and encode it
    candidate_vectors = model.encode(df['semantic_text'].tolist(), show_progress_bar=True)

    # 4. Create a Job Description Vector
    # We need to turn the JD into a vector so we can compare it to the candidates.
    print("4️⃣ Encoding the Job Description...")
    jd_text = """
    Deep technical depth in modern ML systems embeddings, retrieval, ranking, LLMs, fine-tuning. 
    Production experience with embeddings-based retrieval systems sentence-transformers vector databases hybrid search.
    Strong Python code quality. Hands-on experience designing evaluation frameworks for ranking systems.
    """
    jd_vector = model.encode([jd_text])

    # 5. Save the math to files
    print("5️⃣ Saving vectors to data/ folder...")
    np.save('../data/candidate_vectors.npy', candidate_vectors)
    np.save('../data/jd_vector.npy', jd_vector)
    
    print("✅ Pro-Level Embeddings Complete! Vectors saved.")

if __name__ == "__main__":
    create_embeddings()