"""
Parses timestamped expert-call transcripts into structured, citable segments.

Each transcript follows the pattern:
    MM:SS
    Speaker: text...

We parse this into a list of segments, each with:
    - timestamp (str, e.g. "02:18")
    - speaker (str)
    - text (str)
    - transcript_id (str)
    - expert_name (str)
    - role (str)
    - market (str)

Grounding is the whole point of this case: every downstream answer must be
traceable back to one of these segments. Nothing here invents text.
"""

import re
import os
import json


def parse_transcript(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = raw.strip().split("\n")

    # Header: first three non-empty lines are Expert / Role / Market
    header_lines = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if re.match(r"^\d{2}:\d{2}$", line.strip()):
            body_start = i
            break
        header_lines.append(line.strip())

    expert_name = header_lines[0].split("–")[-1].strip() if header_lines else "Unknown"
    role = header_lines[1].replace("Role:", "").strip() if len(header_lines) > 1 else ""
    market = header_lines[2].replace("Market:", "").strip() if len(header_lines) > 2 else ""

    body = "\n".join(lines[body_start:])

    # Split on timestamp markers like "00:18"
    pattern = r"(\d{2}:\d{2})\n"
    parts = re.split(pattern, body)

    segments = []
    # parts alternates: ['', '00:00', 'Interviewer: ...\n\n', '00:18', 'Dr. Martin: ...\n\n', ...]
    for i in range(1, len(parts), 2):
        timestamp = parts[i].strip()
        block = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # A block may contain one or more "Speaker: text" lines
        speaker_match = re.match(r"^([^:]+):\s*(.*)", block, re.DOTALL)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            text = speaker_match.group(2).strip().replace("\n", " ")
        else:
            speaker = "Unknown"
            text = block.replace("\n", " ")

        segments.append({
            "timestamp": timestamp,
            "speaker": speaker,
            "text": text,
        })

    transcript_id = os.path.basename(filepath).replace(".txt", "")

    return {
        "transcript_id": transcript_id,
        "expert_name": expert_name,
        "role": role,
        "market": market,
        "segments": segments,
    }


def build_corpus(data_dir: str) -> list:
    """Parse all transcripts in a directory into a flat list of citable chunks.

    Each chunk pairs an interviewer question with the expert's answer that
    immediately follows it, since that pairing is what actually answers the
    interview guide questions and is what a user will want cited together.
    """
    transcripts = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith("Transcript") and fname.endswith(".txt"):
            transcripts.append(parse_transcript(os.path.join(data_dir, fname)))

    chunks = []
    for t in transcripts:
        segs = t["segments"]
        i = 0
        while i < len(segs):
            seg = segs[i]
            if seg["speaker"].lower() in ("interviewer",):
                question = seg["text"]
                q_ts = seg["timestamp"]
                # pair with next non-interviewer segment (the answer)
                if i + 1 < len(segs):
                    ans = segs[i + 1]
                    chunks.append({
                        "transcript_id": t["transcript_id"],
                        "expert_name": t["expert_name"],
                        "role": t["role"],
                        "market": t["market"],
                        "question": question,
                        "question_timestamp": q_ts,
                        "answer": ans["text"],
                        "answer_timestamp": ans["timestamp"],
                        "speaker": ans["speaker"],
                        # searchable text combines Q+A so retrieval matches on topic, not just answer wording
                        "search_text": f"{question} {ans['text']}",
                    })
                    i += 2
                    continue
            i += 1

    return transcripts, chunks


if __name__ == "__main__":
    transcripts, chunks = build_corpus(os.path.dirname(__file__))
    print(f"Parsed {len(transcripts)} transcripts, {len(chunks)} Q&A chunks total.\n")
    for c in chunks[:3]:
        print(json.dumps(c, indent=2))
        print("---")
