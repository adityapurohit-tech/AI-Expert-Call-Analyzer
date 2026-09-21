"""
LLM layer: takes retrieved, citable chunks and produces grounded output.

The single most important design decision in this file is the prompt
contract, not the model choice:

  - The model is given ONLY the retrieved chunks as context.
  - It is instructed to answer strictly from that context.
  - It must cite (expert, market, timestamp) for every claim.
  - If the context doesn't support an answer, it must say so explicitly
    rather than filling the gap with plausible-sounding text.

This is the same pattern used throughout: retrieval scopes what the model
is allowed to say, and the prompt forces it to show its work. Hallucination
reduction here isn't a separate feature bolted on afterward — it's the
structure of the prompt itself.

Model choice: Llama, run fully locally via Ollama. Three deliberate reasons,
not just convenience:
  1. Zero external API dependency and zero per-call cost.
  2. Data privacy — these are confidential expert-call transcripts. Nothing
     ever leaves the machine, which matters more here than for a toy demo.
  3. The architecture is intentionally model-agnostic: every model call
     goes through one function, _call_model() below. Swapping to Groq,
     Claude, or GPT is a one-function change; nothing in the retrieval or
     prompting logic changes. This mirrors how I'd approach model selection
     in production — pick on capability, cost, latency, and (here) data
     sensitivity for the specific job, not vendor loyalty.
"""

import os
import requests

# Configurable so this points at whatever tag you actually have pulled locally.
# Check with: ollama list
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

SYSTEM_PROMPT = """You are analysing expert interview transcripts for a market research project.

STRICT RULES:
1. Only use information from the provided transcript excerpts. Never use outside knowledge about robotic surgery, healthcare markets, or any other topic.
2. Every factual claim must be attributable to a specific expert and timestamp from the excerpts given to you.
3. If the excerpts do not contain enough information to answer, say so explicitly. Do not guess or fill gaps.
4. When quoting, use the expert's exact words. Do not paraphrase and present it as a quote.
5. Be concise and analytical, not conversational."""


def _call_model(system: str, user: str, max_tokens: int = 400) -> str:
    """Calls a local Ollama server. Requires Ollama to be running
    (`ollama serve`, usually already running in the background after install)
    and the model already pulled (`ollama pull llama3.1` or whichever tag
    you have — set OLLAMA_MODEL to match).
    """
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # low temperature: this task rewards consistency and literalness, not creativity
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Make sure Ollama is running "
            f"locally (`ollama serve`) and that you have pulled a model "
            f"(`ollama pull llama3.1` or set OLLAMA_MODEL to match a model you already have, "
            f"check with `ollama list`)."
        )


def _format_chunks_as_context(chunks: list) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"[{c['expert_name']} | {c['market']} | {c['answer_timestamp']}]\n"
            f"Q: {c['question']}\n"
            f"A: {c['answer']}\n"
        )
    return "\n".join(lines)


def answer_interview_question(question: str, per_expert_chunks: dict) -> dict:
    """Answer one interview-guide question, per expert, with citations.

    per_expert_chunks: {transcript_id: [chunk, ...]} from retriever.search_per_transcript
    Returns: {transcript_id: {"answer": str, "quote": str, "timestamp": str, "confidence": "grounded"|"not_discussed"}}
    """
    results = {}

    for tid, chunks in per_expert_chunks.items():
        if not chunks or chunks[0]["score"] < 0.05:
            results[tid] = {
                "expert_name": chunks[0]["expert_name"] if chunks else tid,
                "answer": "Not clearly discussed in this transcript.",
                "quote": None,
                "timestamp": None,
                "grounded": False,
            }
            continue

        context = _format_chunks_as_context(chunks)
        prompt = f"""Interview guide question: "{question}"

Transcript excerpts for this expert:
{context}

Answer the interview guide question based ONLY on these excerpts, in 1-2 sentences.
Then provide the single most relevant exact quote (verbatim) that supports your answer, with its timestamp.

Respond in this exact format:
ANSWER: <your 1-2 sentence answer>
QUOTE: <exact verbatim quote>
TIMESTAMP: <timestamp>

If the excerpts don't address this question at all, respond with:
ANSWER: Not discussed.
QUOTE: none
TIMESTAMP: none"""

        text = _call_model(SYSTEM_PROMPT, prompt, max_tokens=300)

        answer_line = _extract_field(text, "ANSWER")
        quote_line = _extract_field(text, "QUOTE")
        ts_line = _extract_field(text, "TIMESTAMP")

        results[tid] = {
            "expert_name": chunks[0]["expert_name"],
            "answer": answer_line,
            "quote": quote_line if quote_line and quote_line.lower() != "none" else None,
            "timestamp": ts_line if ts_line and ts_line.lower() != "none" else None,
            "grounded": quote_line is not None and quote_line.lower() != "none",
        }

    return results


def synthesize_themes(question: str, per_expert_answers: dict) -> str:
    """Given per-expert answers to the same question, identify common
    themes and explicit disagreements. This is the one place we ask the
    model to compare rather than just extract, so the prompt is explicit
    that comparison must still be grounded in the stated answers only.
    """
    answers_block = "\n\n".join(
        f"{data['expert_name']}: {data['answer']}"
        + (f' (quote: "{data["quote"]}")' if data.get("quote") else "")
        for data in per_expert_answers.values()
    )

    prompt = f"""Interview guide question: "{question}"

Here is what each expert said in response (already extracted from their transcripts):

{answers_block}

Identify:
1. Common themes — points where 2 or more experts agree or say similar things.
2. Disagreements — points where experts give notably different views, numbers, or emphasis.

Base this ONLY on the answers given above. Do not introduce outside information. Be specific about which expert said what."""

    return _call_model(SYSTEM_PROMPT, prompt, max_tokens=400)


def answer_free_question(question: str, retrieved_chunks: list) -> dict:
    """Free-form cross-transcript Q&A, grounded in whatever the retriever
    surfaced for this specific question.
    """
    if not retrieved_chunks or retrieved_chunks[0]["score"] < 0.03:
        return {
            "answer": "This doesn't appear to be discussed in the transcripts.",
            "sources": [],
        }

    context = _format_chunks_as_context(retrieved_chunks)
    prompt = f"""User question: "{question}"

Relevant transcript excerpts:
{context}

Answer the user's question based ONLY on these excerpts. Cite which expert(s) and timestamp(s) support your answer.
If the excerpts don't fully answer the question, say what is and isn't covered."""

    answer_text = _call_model(SYSTEM_PROMPT, prompt, max_tokens=500)

    sources = [
        {"expert": c["expert_name"], "market": c["market"], "timestamp": c["answer_timestamp"]}
        for c in retrieved_chunks
    ]
    return {"answer": answer_text, "sources": sources}


def _extract_field(text: str, field: str) -> str:
    for line in text.split("\n"):
        if line.strip().upper().startswith(field.upper() + ":"):
            return line.split(":", 1)[1].strip()
    return ""
