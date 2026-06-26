import pandas as pd
import os

def generate_submission():
    input_path = '../data/final_ranked_candidates.csv'
    
    # The rules state the filename must be your team ID. 
    # Change "team_redrob" to whatever your registered team name is!
    output_path = '../team_redrob.csv' 
    
    if not os.path.exists(input_path):
        print("❌ Error: final_ranked_candidates.csv not found.")
        return

    print("1️⃣ Loading ranked candidate data...")
    df = pd.read_csv(input_path)
    
    # --- RULE 1: EXACTLY 100 CANDIDATES ---
    # The validator will fail if you submit more than 100.
    df = df.head(100).copy()
    
    # --- RULE 2: THE REASONING COLUMN ---
    # The judges want a 1-2 sentence explanation for why they got this rank. 
    # We will programmatically generate this using the math we already calculated!
    print("2️⃣ Generating AI Reasoning strings...")
    def create_reasoning(row):
        sem_score = round(row['semantic_score'] * 100, 1)
        resp_rate = round(row['response_rate'] * 100, 1)
        return f"Selected for a {sem_score}% semantic alignment with the JD, supported by a strong {resp_rate}% recruiter response rate."
    
    df['reasoning'] = df.apply(create_reasoning, axis=1)
    
    # --- RULE 3: EXACT COLUMNS ---
    print("3️⃣ Formatting columns to pass the auto-validator...")
    df = df.rename(columns={'final_score': 'score'})
    
    # Select ONLY the 4 required columns in the exact order requested
    submission_df = df[['candidate_id', 'rank', 'score', 'reasoning']]
    
    # Save it directly to the root folder so it's easy to upload
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Official submission file created successfully: {output_path}")
    print("-" * 50)
    print("Preview of your final upload:")
    print(submission_df.head(3).to_string(index=False))

if __name__ == "__main__":
    generate_submission()