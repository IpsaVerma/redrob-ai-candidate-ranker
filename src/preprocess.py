import json
import pandas as pd
import re
import os

def clean_text(text):
    """Pro-Tip: Always sanitize text before feeding it to an AI model."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep only letters, numbers, and spaces to remove weird formatting
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse multiple spaces into one
    return re.sub(r'\s+', ' ', text).strip()

def process_candidates(json_file_path):
    print(f"Loading data from {json_file_path}...")
    
    with open(json_file_path, 'r') as f:
        candidates = json.load(f)
    
    processed_list = []
    
    # Loop through every candidate in the JSON file
    for cand in candidates:
        cand_id = cand['candidate_id']
        profile = cand.get('profile', {})
        
        # --- HOW TO BUILD THE "SEMANTIC DOCUMENT" ---
        summary = profile.get('summary', '')
        
        # Loop through their job history and extract descriptions
        exp_descriptions = []
        for job in cand.get('career_history', []):
            if job.get('description'):
                exp_descriptions.append(job['description'])
        full_experience = " ".join(exp_descriptions)
        
        # Grab their skills
        skills = " ".join([s['name'] for s in cand.get('skills', [])])
        
        # Combine everything into one giant paragraph for the AI to read
        raw_text = f"Title: {profile.get('current_title', '')}. Summary: {summary} Experience: {full_experience} Skills: {skills}"
        cleaned_doc = clean_text(raw_text)
        
        # --- EXTRACTING THE SECRET SAUCE (BEHAVIORAL SIGNALS) ---
        signals = cand.get('redrob_signals', {})
        
        processed_list.append({
            'candidate_id': cand_id,
            'name': profile.get('anonymized_name', 'Unknown'),
            'semantic_text': cleaned_doc,
            'response_rate': signals.get('recruiter_response_rate', 0.0),
            'interview_rate': signals.get('interview_completion_rate', 0.0),
            'open_to_work': int(signals.get('open_to_work_flag', False))
        })
        
    # Convert our list of dictionaries into a clean Pandas DataFrame
    df = pd.DataFrame(processed_list)
    return df

if __name__ == "__main__":
    # Ensure the file path is correct relative to where we run the script
    file_path = '../data/sample_candidates.json'
    
    if os.path.exists(file_path):
        df = process_candidates(file_path)
        print("\n✅ Pro-Level Parsing Complete!")
        print(f"Total Candidates Processed: {len(df)}")
        
        # Save this clean data to a CSV so the next steps are easier
        df.to_csv('../data/processed_candidates.csv', index=False)
        print("Saved cleaned data to: data/processed_candidates.csv")
        
        print("\nHere is what the AI will actually read (First Candidate):")
        print(df['semantic_text'].iloc[0][:200] + "...")
    else:
        print(f"❌ Error: Cannot find {file_path}. Make sure the file is in the data folder!")