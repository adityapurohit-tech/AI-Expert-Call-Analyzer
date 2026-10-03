# Expert Call Analyzer — Case Study

A small application that analyses 3 expert-call transcripts (robotic surgery adoption in France, Germany, and the UK), answers a structured interview guide with citations, identifies cross-expert themes and disagreements, and supports free-form Q&A — all grounded strictly in the source transcripts.

## What it does

1. **Interview Guide tab** — for each of the 6 guide questions, shows each expert's answer alongside the exact quote and timestamp it came from.
2. **Themes & Disagreements tab** — compares the three experts' answers to a given question and surfaces where they agree and where they genuinely diverge.
3. **Ask a Question tab** — free-form Q&A across all three transcripts, with citations, and an explicit "not discussed" response when the transcripts don't cover something.
4. **Raw Transcripts tab** — the parsed transcripts with timestamps, so every AI-generated claim can be manually checked against the source in one click.

## Architecture

```
data/                      3 transcripts + interview guide (as provided)
app/
  parser.py                Parses timestamped transcripts into citable (question, answer, timestamp, expert) chunks
  retriever.py             TF-IDF + cosine similarity search over those chunks
  llm.py                   Claude calls with a strict grounding prompt contract
  app.py                   Streamlit UI wiring it together
```

**Pipeline for every question, guide-based or free-form:**

`question → retrieve top-matching chunks (per expert or across all) → pass ONLY those chunks to Claude as context → Claude answers strictly from that context, citing expert + timestamp`

The model is never given the full transcripts and never given general knowledge about robotic surgery. It only ever sees the specific excerpts retrieval selected for that question. This is what makes every answer traceable — if you follow the citation, you get to the exact sentence that produced the answer.

## Model choice

**Llama, run fully locally via Ollama** — no cloud API at all. Three deliberate reasons, not just convenience:

1. **Zero external dependency and zero per-call cost.** The whole pipeline runs offline once the model is pulled.
2. **Data privacy.** These are confidential expert-call transcripts. With a local model, the transcript content never leaves the machine — nothing is sent to a third-party API. For real client research data, this is a genuinely material consideration, not a nice-to-have.
3. **The architecture is intentionally model-agnostic.** Every model call goes through one function, `_call_model()` in `llm.py`. Swapping to a hosted model (Groq, Claude, GPT, Gemini) is a one-function change; nothing else in the retrieval or prompting pipeline changes. This mirrors how I'd approach model selection in production: pick on capability, cost, latency, and — here specifically — data sensitivity, rather than defaulting to whichever vendor is most hyped, and keep the door open to re-test that choice as requirements shift.

## Prerequisite: Ollama running locally

1. Install Ollama if you don't already have it: `ollama.com`
2. Pull a Llama model if you haven't: `ollama pull llama3.1`
3. Confirm what you have available: `ollama list`
4. If your tag isn't `llama3.1`, set it before running the app: `export OLLAMA_MODEL=<your-tag>`

## How citations/timestamps work

Transcripts are parsed into **question–answer pairs**, not arbitrary text chunks. Each pair carries its own timestamp and speaker. This matters because:
- A generic chunking strategy (e.g., fixed 200-character windows) would frequently split a question from its answer, or split one answer across two chunks — breaking the very citation this case requires.
- Pairing Q&A means every retrieved unit is already a complete, self-contained, citable fact.

The LLM is instructed to quote **verbatim** and never paraphrase text presented as a quote, and every UI element that shows a quote shows its timestamp directly beside it.

## How hallucination is reduced

This was the design constraint I treated as non-negotiable, not an add-on:

1. **Retrieval scopes the model's world.** Claude only ever sees the chunks retrieval selected — never the full transcript, never outside knowledge.
2. **The system prompt explicitly forbids outside knowledge** about robotic surgery or healthcare markets, and requires every claim to cite an expert + timestamp.
3. **A similarity threshold gates generation.** If the top retrieved match scores below a minimum relevance threshold, the app shows "Not discussed" *before* calling the LLM at all, rather than asking the model to make the best of weak context.
4. **Verbatim quote requirement.** The model is told to quote exactly or not at all — this is checked structurally (the QUOTE field must be literal transcript text), not just requested politely.
5. **The Raw Transcripts tab exists specifically so every claim can be manually spot-checked** against source in one click — trust is designed to be earned by verification, not just asserted.

**Known limitation, found during testing:** with only 3 short transcripts, TF-IDF retrieval can occasionally rank a topically-adjacent answer (e.g., "surgeon training") above the more directly relevant one (e.g., "cost/barriers") for a query like "how important are budgets and ROI," because of keyword overlap in short text. The app compensates by passing the top 2 candidates per expert to Claude rather than just the top 1, so the LLM can select the more relevant one from real alternatives rather than being locked into a single retrieval mistake. At the current 3-transcript scale this fully resolves the issue in practice; the section below explains why this wouldn't hold at 30+ transcripts.

## How this would scale from 3 transcripts to 30+

At 3 transcripts, TF-IDF is fast, transparent, and sufficient. It would **not** hold up at 30+ transcripts, for concrete reasons:

- **Retrieval quality**: TF-IDF matches on keyword overlap, not meaning. At 30+ transcripts the corpus becomes topically dense enough that keyword collisions (like the training/cost example above) would happen often enough to matter. I would move to embedding-based retrieval (e.g., `sentence-transformers` or an API embedding model) with a proper vector store (FAISS or a managed vector DB), which retrieves on semantic similarity rather than surface word overlap.
- **Cross-expert synthesis**: right now, `synthesize_themes()` passes all expert answers directly into one prompt. At 30+ experts that exceeds a reasonable context window and gets expensive per call. I would batch this — cluster similar answers first (e.g., via embedding similarity or a cheap classification pass), then synthesize per cluster, then do a final roll-up synthesis across cluster summaries.
- **Evaluation**: at 3 transcripts I can eyeball every answer against the source. At 30+, that's not possible. I would build a small labeled evaluation set (a sample of question/expert pairs with a human-verified correct answer + quote) and run it after every change to catch regressions in retrieval or generation quality before they ship — the same evaluation-first instinct I've applied to production RAG work.
- **UI**: the Interview Guide tab currently shows one column per expert. At 30+ experts that needs to become a searchable, filterable table with drill-down, not a fixed grid of columns.

## Running it locally

```bash
# Ollama must already be running with a Llama model pulled — see prerequisite above.
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. The Interview Guide, Themes, and Ask-a-Question tabs require Ollama to be running; the Raw Transcripts tab works without it.

## What I would build next, given more time

- Confidence scoring surfaced in the UI (not just a binary grounded/not-grounded), so a user can see *how* confident the retrieval match was
- A "compare all 6 guide questions at once" export view (e.g., a downloadable summary table) for someone who wants the full picture without clicking through each tab
- Basic caching so re-asking an already-analyzed question doesn't re-call the API
