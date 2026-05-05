"""
1_extract_and_chunk.py
─────────────────────────────
Extracts text from adolescent health PDFs, filters noise,
chunks by sentences, and tags with metadata for RAG.
"""

import fitz  # PyMuPDF
import re
import json
import argparse
import os
from pathlib import Path

# --- 1. FRONT MATTER DETECTION ---
FRONT_MATTER_SIGNALS = [
    r"^(acknowledgements?|foreword|preface|copyright|disclaimer|table of contents?|"
    r"list of (figures|tables|abbreviations)|about this (report|document|manual)|"
    r"executive summary|abbreviations and acronyms|how to use this)",
]

REAL_CONTENT_SIGNALS = [
    r"^(chapter|module|section|part|unit)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    r"^introduction\b", r"^background\b", r"^overview\b",
]

def detect_front_matter_end(doc, max_scan=40):
    last_front_matter_page = 0
    for i, page in enumerate(doc):
        if i >= max_scan: break
        text = page.get_text("text").strip()
        if not text:
            last_front_matter_page = i
            continue

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        first_line = lines[0].lower() if lines else ""
        all_text_lower = text.lower()

        is_roman = bool(re.match(r"^(i{1,3}|iv|v|vi{1,3}|ix|x)\s*$", first_line))
        is_short = len(text) < 300

        has_front_signal = any(re.search(p, all_text_lower, re.MULTILINE) for p in FRONT_MATTER_SIGNALS)
        has_content_signal = any(re.search(p, all_text_lower, re.MULTILINE) for p in REAL_CONTENT_SIGNALS)

        if has_content_signal and not has_front_signal and not is_roman and not is_short:
            if len(text) > 400: return i

        if is_roman or is_short or has_front_signal:
            last_front_matter_page = i

    return last_front_matter_page + 1

# --- 2. HEADING DETECTION ---
HEADING_PATTERNS = [
    (r"^MODULE\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+)\b.*", "module"),
    (r"^CHAPTER\s+(\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b.*", "chapter"),
    (r"^SECTION\s+(\w+)[:\s].*", "section"),
    (r"^SESSION\s+\d+\s*[:\-].*", "session"),
]

def extract_heading(line):
    line = line.strip()
    for pattern, level in HEADING_PATTERNS:
        if re.match(pattern, line, re.IGNORECASE):
            return (level, line)
    return None

# --- 3. SAFETY & RELEVANCE FILTERING ---
DROP_PATTERNS = [
    r"\b(p\s*[<=>]\s*0\.\d+)\b",  # p-values
    r"\b\d{4,}\b",                # large statistics
    r"(see table \d+|refer to figure \d+|appendix \w+)",
    r"(ibid|et al|doi:|isbn:|issn:)",
    r"^\s*(references?|bibliography)\s*$",
    r"www\.|http[s]?://",
    r"©|all rights reserved",
]

KEEP_SIGNALS = [
    r"\b(is|are|means?|refers? to|defined? as|involves?|describes?)\b",
    r"\b(can cause|may lead to|results? in|affects?)\b",
    r"\b(adolescent|young people|youth|teenager)\b",
    r"\b(health|mental|sexual|nutrition|violence|substance)\b",
]

def is_useful_chunk(text):
    if len(text.split()) < 20: return False
    text_lower = text.lower()

    noise_count = sum(1 for p in DROP_PATTERNS if re.search(p, text_lower))
    if noise_count >= 2: return False

    return any(re.search(p, text_lower) for p in KEEP_SIGNALS)

# --- 4. CATEGORIZATION & INTENT ---
TOPIC_KEYWORDS = {
    "mental_health": ["mental health", "depression", "anxiety", "suicide", "stress"],
    "sexual_health": ["sexual", "reproductive", "sti", "hiv", "pregnancy", "puberty"],
    "nutrition": ["nutrition", "diet", "food", "malnutrition", "obesity"],
    "substance_abuse": ["drug", "alcohol", "tobacco", "addiction", "abuse"],
    "violence": ["violence", "assault", "rape", "bullying", "injury"],
    "chronic_disease": ["diabetes", "asthma", "sickle cell", "seizure", "chronic"],
    "general_health": ["adolescent", "youth", "development", "health"],
}

def categorize(text):
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_health"

def label_intent(text):
    text_lower = text.lower()
    if any(s in text_lower for s in ["suicide", "self-harm", "rape", "emergency", "danger"]): return "crisis"
    if any(s in text_lower for s in ["dosage", "prescribe", "diagnose", "treatment protocol", "clinical"]): return "clinical"
    if any(s in text_lower for s in ["is defined as", "refers to", "means", "is a condition"]): return "descriptive"
    return "awareness"

# --- 5. SENTENCE-AWARE CHUNKING ---
def chunk_text(text, target_words=80, overlap_sentences=1):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    chunks, current_chunk, current_word_count = [], [], 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_word_count + sentence_words > target_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_word_count = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sentence)
        current_word_count += sentence_words

    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

# --- 6. MAIN EXTRACTION ---
def extract_for_rag(pdf_path, source_name, chunk_size, overlap):
    doc = fitz.open(pdf_path)
    print(f"\n📄 Document: {source_name} | Total pages: {len(doc)}")

    content_start = detect_front_matter_end(doc)
    print(f"   Real content starts at page index: {content_start}")

    current_module, current_session = "General", "General"
    all_chunks, chunk_counter = [], 0

    for page_num in range(content_start, len(doc)):
        text = doc[page_num].get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in lines[:10]:
            heading = extract_heading(line)
            if heading:
                level, h_text = heading
                if level in ("module", "chapter"): current_module = h_text
                elif level in ("session", "section"): current_session = h_text

        clean_text = " ".join(text.split())
        page_chunks = chunk_text(clean_text, chunk_size, overlap)

        for chunk in page_chunks:
            if is_useful_chunk(chunk):
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"{source_name}_chunk_{chunk_counter:04d}",
                    "source": source_name,
                    "module": current_module,
                    "session": current_session,
                    "category": categorize(chunk),
                    "intent": label_intent(chunk),
                    "page_number": page_num + 1,
                    "text": chunk,
                })

    print(f"✅ Useful chunks extracted: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDFs to RAG-ready JSON")
    parser.add_argument("pdf_path", help="Path to the PDF document")
    parser.add_argument("--source", type=str, required=True, help="Source name (e.g., Nigeria_Manual)")

    args = parser.parse_args()

    chunks = extract_for_rag(args.pdf_path, args.source, 80, 1)

    out_path = f"{args.source}_rag_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to {out_path}")