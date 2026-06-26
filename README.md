# 🎯 Redrob AI Candidate Ranker

An intelligent candidate discovery and ranking system built for the Redrob Data & AI Challenge. This system moves beyond traditional keyword matching by implementing a **Hybrid Semantic & Behavioral Scoring Engine**.

---

## 🧠 Architecture

Our system ranks candidates by reasoning about the gap between **what the Job Description says and what the Job Description actually means**, while heavily considering behavioral availability.

### Pipeline

1. **Semantic Document Generation**

   * Parse complex nested candidate JSON profiles
   * Extract skills, summaries, and career histories
   * Convert them into unified semantic documents

2. **High-Dimensional Embeddings**

   * Utilize `sentence-transformers/all-MiniLM-L6-v2`
   * Generate 384-dimensional vectors for:

     * Candidate profiles
     * Job descriptions

3. **Cosine Similarity Scoring**

   * Measure semantic similarity between candidate vectors and JD vectors
   * Avoid traditional keyword stuffing issues

4. **Behavioral Penalty Multipliers**

Final Hybrid Score Formula:

```text
Final Score =
(0.60 × Semantic Score)
+ (0.20 × Response Rate)
+ (0.10 × Interview Rate)
+ (0.10 × Open To Work)
```

---

## 📂 Project Structure

```text
challenge_1
│
├── data/
│   ├── raw datasets
│   ├── processed_candidates.csv
│   ├── candidate_vectors.npy
│   ├── jd_vector.npy
│   └── final_ranked_candidates.csv
│
├── src/
│   ├── preprocess.py
│   ├── embeddings.py
│   ├── scorer.py
│   └── ranker.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
cd src

python preprocess.py
python embeddings.py
python scorer.py
python ranker.py
```

Launch Streamlit dashboard:

```bash
cd ..

streamlit run app.py
```

---

## 📊 Features

✅ Semantic candidate matching

✅ Hybrid scoring engine

✅ Behavioral signal analysis

✅ Candidate ranking dashboard

✅ Search functionality

✅ Download ranked candidate results

✅ Top candidate visualizations

---

## 📈 Output

The system generates:

* `processed_candidates.csv`
* `candidate_vectors.npy`
* `final_ranked_candidates.csv`
* Interactive Streamlit dashboard

---

## 🔮 Future Scope

* Resume PDF parsing
* Real-time recruiter dashboards
* Explainable AI recommendations
* Multi-job candidate matching
