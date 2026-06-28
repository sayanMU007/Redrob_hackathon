# app.py
import streamlit as st
import json
import csv
import io
import sys
sys.path.insert(0, ".")
from rank import rank_candidates

st.title("Redrob Candidate Ranker")
st.write("Upload a candidates JSONL file to get a ranked CSV of top candidates.")

uploaded_file = st.file_uploader("Upload candidates.jsonl", type=["jsonl", "json"])

if uploaded_file is not None:
    # Read candidates
    candidates = []
    content = uploaded_file.read().decode("utf-8")
    for line in content.strip().split("\n"):
        if line.strip():
            candidates.append(json.loads(line))

    st.write(f"Loaded **{len(candidates)}** candidates")

    if st.button("Rank Candidates"):
        with st.spinner("Ranking..."):
            ranked = rank_candidates(candidates)
            top_n = min(100, len(ranked))
            ranked = ranked[:top_n]

        st.success(f"Done! Ranked {top_n} candidates.")

        # Show top 10 in UI
        st.subheader("Top 10 Preview")
        for i, (cid, score, reasoning) in enumerate(ranked[:10], 1):
            st.write(f"**{i}.** {cid} — score: {score:.4f}")
            st.caption(reasoning[:150])

        # Generate downloadable CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning) in enumerate(ranked, 1):
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

        st.download_button(
            label="Download ranked CSV",
            data=output.getvalue(),
            file_name="submission.csv",
            mime="text/csv"
        )