"""
Streamlit UI for the Hasamex AI Engineer case study.

Three views, matching the three things the brief asks the app to do:
  1. Interview Guide tab  -> per-expert answers + quotes + timestamps
  2. Themes tab           -> cross-expert synthesis (agreement / disagreement)
  3. Ask a Question tab   -> free-form Q&A across all transcripts

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from parser import build_corpus
from retriever import TranscriptRetriever
from llm import answer_interview_question, synthesize_themes, answer_free_question

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GUIDE_QUESTIONS = [
    "How would you describe current adoption of robotic surgery in your market?",
    "What are the main barriers to adoption?",
    "How important are hospital budgets and ROI in purchasing decisions?",
    "How important are surgeon training and clinical outcomes?",
    "What adoption trend do you expect over the next 3–5 years?",
    "What is the typical hospital decision-making timeline for purchasing a new robotic system?",
]

st.set_page_config(page_title="Expert Call Analyzer", layout="wide")


@st.cache_resource
def load_corpus():
    transcripts, chunks = build_corpus(DATA_DIR)
    retriever = TranscriptRetriever(chunks)
    return transcripts, chunks, retriever


transcripts, chunks, retriever = load_corpus()
transcript_ids = [t["transcript_id"] for t in transcripts]

st.title("Expert Call Analyzer")
st.caption(
    f"{len(transcripts)} transcripts loaded · "
    + " · ".join(f"{t['expert_name']} ({t['market']})" for t in transcripts)
)

st.info(
    "This app calls a local Ollama server for AI features. Make sure Ollama is "
    "running (`ollama serve`, usually auto-started) and you have a model pulled "
    "(check with `ollama list`). If your local tag isn't `llama3.1`, set "
    "`OLLAMA_MODEL=<your-tag>` before launching. The Raw Transcripts tab works "
    "even if Ollama isn't running."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Interview Guide", "🔍 Themes & Disagreements", "💬 Ask a Question", "📄 Raw Transcripts"]
)

# ---------- TAB 1: Interview Guide ----------
with tab1:
    st.subheader("Interview Guide Answers")
    st.caption("Each answer is generated only from that expert's transcript, with an exact supporting quote and timestamp.")

    question = st.selectbox("Select an interview guide question:", GUIDE_QUESTIONS, key="guide_q")

    if st.button("Analyze this question", key="analyze_guide"):
        with st.spinner("Retrieving relevant excerpts and generating grounded answers..."):
            per_expert_chunks = retriever.search_per_transcript(question, transcript_ids, top_k_each=2)
            answers = answer_interview_question(question, per_expert_chunks)
            st.session_state["last_answers"] = answers
            st.session_state["last_question"] = question

    if "last_answers" in st.session_state and st.session_state.get("last_question") == question:
        answers = st.session_state["last_answers"]
        cols = st.columns(len(answers))
        for col, (tid, data) in zip(cols, answers.items()):
            with col:
                st.markdown(f"**{data['expert_name']}**")
                st.write(data["answer"])
                if data["grounded"]:
                    st.markdown(f"> *\"{data['quote']}\"*")
                    st.caption(f"⏱ {data['timestamp']}")
                else:
                    st.caption("No direct answer found in this transcript.")

# ---------- TAB 2: Themes & Disagreements ----------
with tab2:
    st.subheader("Cross-Expert Themes & Disagreements")
    st.caption("Synthesized only from the extracted answers shown in the Interview Guide tab — not from re-reading full transcripts, to keep the comparison traceable.")

    theme_question = st.selectbox("Select a question to compare across experts:", GUIDE_QUESTIONS, key="theme_q")

    if st.button("Compare across experts", key="analyze_themes"):
        with st.spinner("Retrieving each expert's answer and identifying themes..."):
            per_expert_chunks = retriever.search_per_transcript(theme_question, transcript_ids, top_k_each=2)
            answers = answer_interview_question(theme_question, per_expert_chunks)
            synthesis = synthesize_themes(theme_question, answers)
            st.session_state["last_synthesis"] = synthesis
            st.session_state["last_theme_question"] = theme_question

    if "last_synthesis" in st.session_state and st.session_state.get("last_theme_question") == theme_question:
        st.markdown(st.session_state["last_synthesis"])

# ---------- TAB 3: Ask a Question ----------
with tab3:
    st.subheader("Ask a Question Across All Transcripts")
    st.caption("Answers are grounded in retrieved excerpts only. If it's not in the transcripts, the app will say so rather than guessing.")

    user_question = st.text_input("Your question:", placeholder="e.g. Did any expert mention specific procedure volume numbers?")

    if st.button("Ask") and user_question.strip():
        with st.spinner("Searching transcripts and generating answer..."):
            results = retriever.search(user_question, top_k=5)
            response = answer_free_question(user_question, results)

        st.markdown(response["answer"])
        if response["sources"]:
            st.markdown("**Sources:**")
            for s in response["sources"]:
                st.caption(f"— {s['expert']} ({s['market']}), {s['timestamp']}")

# ---------- TAB 4: Raw Transcripts ----------
with tab4:
    st.subheader("Raw Transcripts (for verification)")
    selected = st.selectbox("View transcript:", transcript_ids, key="raw_view")
    t = next(t for t in transcripts if t["transcript_id"] == selected)
    st.markdown(f"**{t['expert_name']}** — {t['role']}, {t['market']}")
    for seg in t["segments"]:
        st.markdown(f"`{seg['timestamp']}` **{seg['speaker']}:** {seg['text']}")
