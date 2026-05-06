from __future__ import annotations

import re


CITATION_PATTERN = r"\[[A-Z0-9_.-]+-\d{4}-\d{2}-\d{2}-\d{4}\]"

COMPANY_ALIASES = {
    "aapl": "Apple",
    "apple": "Apple",
    "amzn": "Amazon",
    "amazon": "Amazon",
    "msft": "Microsoft",
    "microsoft": "Microsoft",
    "nvda": "NVIDIA",
    "nvidia": "NVIDIA",
    "tsla": "Tesla",
    "tesla": "Tesla",
}


def format_model_answer(answer: str, question: str | None = None) -> str:
    """Normalize model output into compact Markdown without another model call."""
    cleaned = clean_answer_text(answer)
    if not cleaned:
        return cleaned

    scalar_answer = format_scalar_answer(cleaned, question)
    if scalar_answer:
        return scalar_answer

    if has_inline_numbered_list(cleaned):
        return format_inline_numbered_list(cleaned)

    return format_existing_markdown(cleaned)


def clean_answer_text(answer: str) -> str:
    text = str(answer or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"^(assistant|answer)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_scalar_answer(answer: str, question: str | None) -> str | None:
    if not question:
        return None

    normalized_question = question.lower()
    if "total net sales" not in normalized_question:
        return None

    compact = re.sub(r"\s+", " ", answer)
    match = re.search(
        r"\btotal net sales\b\s+(\d{1,3}(?:,\d{3})+|\d+)\b",
        compact,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    amount = match.group(1)
    company = infer_company_name(question)
    subject = f"{company} reported" if company else "The filing reports"
    year_match = re.search(r"\b(20\d{2})\b", question)
    year_phrase = f" for fiscal year {year_match.group(1)}" if year_match else ""
    units = " million" if re.search(r"\b(in millions|millions)\b", compact, flags=re.IGNORECASE) else ""
    citation = nearest_citation(compact, match.end())
    citation_suffix = f" {citation}" if citation else ""
    return f"{subject} total net sales of ${amount}{units}{year_phrase}.{citation_suffix}"


def infer_company_name(question: str) -> str | None:
    lowered = question.lower()
    for alias, company in COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return company
    return None


def nearest_citation(text: str, offset: int) -> str | None:
    after = re.search(CITATION_PATTERN, text[offset : offset + 600])
    if after:
        return after.group(0)
    any_citation = re.search(CITATION_PATTERN, text)
    return any_citation.group(0) if any_citation else None


def has_inline_numbered_list(text: str) -> bool:
    return len(re.findall(r"(?<!\w)\(\d+\)\s+", text)) >= 2


def format_inline_numbered_list(text: str) -> str:
    first_marker = re.search(r"(?<!\w)\(\d+\)\s+", text)
    if not first_marker:
        return text

    intro = text[: first_marker.start()].strip(" :;\n")
    chunks = re.split(r"(?<!\w)(?=\(\d+\)\s+)", text[first_marker.start() :])
    bullets = []
    seen = set()
    for chunk in chunks:
        item = clean_numbered_item(chunk)
        if not item:
            continue
        key = normalize_for_dedupe(item)
        if key in seen:
            continue
        seen.add(key)
        bullets.append(item)

    if not bullets:
        return text

    if intro:
        intro = ensure_intro_punctuation(intro)
        return intro + "\n\n" + "\n".join(f"- {bullet}" for bullet in bullets)
    return "\n".join(f"- {bullet}" for bullet in bullets)


def clean_numbered_item(chunk: str) -> str:
    item = re.sub(r"^\(\d+\)\s*", "", chunk.strip())
    item = re.sub(r"\s+", " ", item).strip(" ;,")
    if not item:
        return ""
    if is_truncated_fragment(item):
        return ""

    citation_suffixes = re.findall(rf"(?:\s*{CITATION_PATTERN})+$", item)
    citation_suffix = citation_suffixes[-1].strip() if citation_suffixes else ""
    if citation_suffix:
        item = item[: -len(citation_suffix)].rstrip()

    if item and item[-1] not in ".!?":
        item += "."
    return f"{item} {citation_suffix}".strip()


def is_truncated_fragment(item: str) -> bool:
    words = re.findall(r"[A-Za-z0-9$%,.-]+", re.sub(CITATION_PATTERN, "", item))
    if item[-1:] in ".!?":
        return False
    return bool(
        len(words) < 5
        and re.search(r"\b(and|for|from|increased|of|or|potential|the|to|with)$", item, flags=re.IGNORECASE)
    )


def normalize_for_dedupe(item: str) -> str:
    without_citations = re.sub(CITATION_PATTERN, "", item)
    return re.sub(r"[^a-z0-9]+", " ", without_citations.lower()).strip()


def ensure_intro_punctuation(intro: str) -> str:
    if intro[-1:] in ":":
        return intro
    if intro[-1:] in ".!?":
        return intro[:-1] + ":"
    return intro + ":"


def format_existing_markdown(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    if any(re.match(r"^[-*]\s+", line) for line in lines):
        return "\n".join(normalize_bullet_line(line) for line in lines if line).strip()
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_bullet_line(line: str) -> str:
    if re.match(r"^[-*]\s+", line):
        body = re.sub(r"^[-*]\s+", "", line).strip()
        return f"- {body}"
    return line
