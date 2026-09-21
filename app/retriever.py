"""
Retrieval engine for the transcript corpus.

Uses TF-IDF + cosine similarity rather than a neural embedding model.
This is a deliberate choice for this case, not a shortcut:

  - The corpus is tiny (21 Q&A chunks across 3 transcripts). A neural
    embedding model adds latency, a large dependency (torch), and zero
    practical retrieval benefit at this scale.
  - TF-IDF is fully transparent: you can inspect exactly which terms
    drove a match, which matters for a case that explicitly cares about
    explainability and not inventing information.
  - The architecture is swappable: at 30+ transcripts (see README scaling
    section) this same interface would be backed by sentence-transformer
    embeddings + a vector store (e.g. FAISS) without changing the calling
    code, because `search()` just returns ranked chunks either way.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TranscriptRetriever:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus_texts = [c["search_text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def search(self, query: str, top_k: int = 5, transcript_id: str = None) -> list:
        """Return top_k chunks most relevant to query, optionally filtered
        to a single transcript. Each result includes a similarity score so
        the caller can apply a 'not discussed' threshold rather than
        forcing a low-confidence match into an answer.
        """
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]

        indexed = list(enumerate(sims))
        if transcript_id:
            indexed = [(i, s) for i, s in indexed if self.chunks[i]["transcript_id"] == transcript_id]

        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in indexed[:top_k]:
            chunk = dict(self.chunks[i])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def search_per_transcript(self, query: str, transcript_ids: list, top_k_each: int = 2) -> dict:
        """Search each transcript independently. Used for interview-guide
        questions, where we want each expert's best answer, not just the
        globally top-ranked chunks (which could all come from one expert).
        """
        return {tid: self.search(query, top_k=top_k_each, transcript_id=tid) for tid in transcript_ids}
