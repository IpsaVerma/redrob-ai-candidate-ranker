import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🎯",
    layout="wide"
)

# ---------------- HEADER ----------------

st.title("🎯 Intelligent Candidate Ranking System")

st.markdown("""
This system ranks candidates based on a **Hybrid Scoring Engine**.

It calculates semantic similarity between candidate profiles and job requirements while considering behavioral signals and overall fit.
""")

st.divider()

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():

    file_path = "data/final_ranked_candidates.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    return None


df = load_data()

# ---------------- MAIN DASHBOARD ----------------

if df is not None:

    # ---------------- TOP INSIGHTS ----------------

    st.subheader("🏆 Top Match Insights")

    col1, col2, col3, col4 = st.columns(4)

    top_candidate = df.iloc[0]

    col1.metric(
        "Candidates Processed",
        len(df)
    )

    col2.metric(
        "Top Candidate",
        top_candidate["name"]
    )

    col3.metric(
        "Hybrid Score",
        f"{top_candidate['final_score']*100:.1f}%"
    )

    col4.metric(
        "Semantic Match",
        f"{top_candidate['semantic_score']*100:.1f}%"
    )

    st.divider()

    # ---------------- SEARCH BOX ----------------

    st.subheader("🔍 Search Candidate")

    search = st.text_input(
        "Enter candidate name"
    )

    filtered_df = df.copy()

    if search:
        filtered_df = df[
            df["name"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # ---------------- TABLE FORMATTING ----------------

    st.subheader("📋 Candidate Leaderboard")

    display_df = filtered_df.copy()

    display_df["final_score"] = (
        display_df["final_score"]*100
    ).round(2).astype(str)+"%"

    display_df["semantic_score"] = (
        display_df["semantic_score"]*100
    ).round(2).astype(str)+"%"

    display_df["response_rate"] = (
        display_df["response_rate"]*100
    ).round(0).astype(str)+"%"

    display_df = display_df.rename(
        columns={
            "rank":"Rank",
            "candidate_id":"Candidate ID",
            "name":"Candidate Name",
            "final_score":"Hybrid Score",
            "semantic_score":"Semantic Match",
            "response_rate":"Response Rate"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------- DOWNLOAD BUTTON ----------------

    st.download_button(
        label="📥 Download Ranked Candidates",
        data=df.to_csv(index=False),
        file_name="final_ranked_candidates.csv",
        mime="text/csv"
    )

    st.divider()

    # ---------------- TOP 5 GRAPH ----------------

    st.subheader("📊 Top 5 Candidate Scores")

    top5 = df.head(5)

    fig, ax = plt.subplots(figsize=(16,8))

    ax.bar(
        top5["name"],
        top5["final_score"]
    )

    ax.set_title("Top 5 Candidate Scores")
    ax.set_xlabel("Candidates")
    ax.set_ylabel("Hybrid Score")

    plt.xticks(rotation=30)

    st.pyplot(fig)

    st.divider()

    # ---------------- EXPLANATION ----------------

    st.subheader("🧠 Why Candidate Ranked #1")

    st.write(
        f"""
### {top_candidate['name']}

This candidate ranked first because:

✅ Strong semantic similarity with Job Description

✅ Better recruiter response rate

✅ Positive behavioral indicators

✅ Highest overall hybrid score

Hybrid Score: **{top_candidate['final_score']*100:.2f}%**

Semantic Match: **{top_candidate['semantic_score']*100:.2f}%**
"""
    )

    st.success(
        "✅ Dashboard loaded successfully!"
    )

else:

    st.error(
        "❌ Could not find final_ranked_candidates.csv"
    )

    st.info(
        "Make sure the file exists in:\n\n data/final_ranked_candidates.csv"
    )
    
st.sidebar.title("Project Details")

st.sidebar.info("""
AI Resume Ranker

Tech Stack:
• Python
• Streamlit
• Pandas
• Sentence Transformers
• NumPy
• Scikit-learn

Method:
Hybrid Ranking System
""")