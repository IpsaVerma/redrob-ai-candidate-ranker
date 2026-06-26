import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def calculate_scores():
    # File paths
    df_path = '../data/processed_candidates.csv'
    cand_vec_path = '../data/candidate_vectors.npy'
    jd_vec_path = '../data/jd_vector.npy'
    output_path = '../data/final_ranked_candidates.csv'

    if not os.path.exists(df_path) or not os.path.exists(cand_vec_path):
        print("❌ Error: Missing data or vectors. Run preprocess.py and embeddings.py first.")
        return

    print("1️⃣ Loading data and mathematical vectors...")
    df = pd.read_csv(df_path)
    cand_vecs = np.load(cand_vec_path)
    jd_vec = np.load(jd_vec_path)

    # --- THE MAGIC TRICK: SEMANTIC MATCHING ---
    print("2️⃣ Calculating Semantic Similarity...")
    # This measures the "angle" between the JD math and the Candidate math
    semantic_scores = cosine_similarity(cand_vecs, jd_vec).flatten()
    
    # Clip scores to be strictly between 0 and 1
    df['semantic_score'] = np.clip(semantic_scores, 0, 1)

    # --- THE HACKATHON SECRET: BEHAVIORAL SCORING ---
    print("3️⃣ Applying Hybrid Scoring Logic...")
    # Fill any missing behavioral data with 0
    df['response_rate'] = df['response_rate'].fillna(0.0)
    df['interview_rate'] = df['interview_rate'].fillna(0.0)
    df['open_to_work'] = df['open_to_work'].fillna(0.0)
    
    # The Custom Formula:
    # 60% weight to their actual skills and experience
    # 20% weight to if they actually reply to recruiters
    # 10% weight to if they show up to interviews
    # 10% weight to if they are actively looking for a job
    df['final_score'] = (
        (df['semantic_score'] * 0.60) + 
        (df['response_rate'] * 0.20) + 
        (df['interview_rate'] * 0.10) + 
        (df['open_to_work'] * 0.10)
    )

    # --- RANKING ---
    print("4️⃣ Ranking Candidates...")
    # Sort from highest score to lowest
    df_ranked = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    df_ranked['rank'] = df_ranked.index + 1

    # Select the columns we actually want to see in the final report
    output_cols = ['rank', 'candidate_id', 'name', 'final_score', 'semantic_score', 'response_rate']
    
    # Save the final results
    df_ranked[output_cols].to_csv(output_path, index=False)
    
    print("\n✅ AI Ranking Complete! Top 5 Candidates:")
    print(df_ranked[output_cols].head(5).to_string(index=False))
    print(f"\nSaved final results to: {output_path}")

if __name__ == "__main__":
    calculate_scores()