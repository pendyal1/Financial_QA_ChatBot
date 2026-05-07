from __future__ import annotations

import re


_CITATION_RE = re.compile(r'\[[A-Z0-9_.-]+-\d{4}-\d{2}-\d{2}-\d{4}\]')
_INLINE_ITEM_RE = re.compile(r'\((\d+)\)\s+')
_TICKER_FROM_CITATION_RE = re.compile(r'\[([A-Z]+)-\d{4}-')

_TICKER_TO_COMPANY: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
    "META": "Meta",
}

_NET_SALES_RE = re.compile(
    r'Total net sales\s*\$?\s*([0-9][0-9,]+)',
    re.IGNORECASE,
)


def _strip_role_prefix(text: str) -> str:
    return re.sub(r'^(assistant|answer)\s*:\s*', '', text, flags=re.IGNORECASE).strip()


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _convert_inline_numbered_list(text: str) -> str:
    """Convert '... (1) foo, (2) bar' into Markdown bullets."""
    parts = _INLINE_ITEM_RE.split(text)
    if len(parts) < 5:
        return text
    preamble = parts[0].rstrip(': ').strip()
    bullets = []
    for i in range(2, len(parts), 2):
        content = parts[i].strip().rstrip(',').rstrip(';').strip()
        if content:
            bullets.append(f"- {content}")
    if not bullets:
        return text
    return (preamble + '\n' if preamble else '') + '\n'.join(bullets)


def _deduplicate_bullets(text: str) -> str:
    seen: set[str] = set()
    deduped = []
    for line in text.split('\n'):
        key = re.sub(r'\s+', ' ', line.strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    return '\n'.join(deduped)


def _drop_truncated_fragments(text: str) -> str:
    filtered = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            filtered.append(line)
            continue
        is_short = len(stripped) < 40
        has_terminal = bool(re.search(r'[.!?]$', stripped))
        has_citation = bool(_CITATION_RE.search(stripped))
        is_bullet = stripped.startswith('-')
        if is_short and not has_terminal and not has_citation and not is_bullet:
            continue
        filtered.append(line)
    return '\n'.join(filtered)


def _extract_total_net_sales(answer: str) -> str | None:
    m = _NET_SALES_RE.search(answer)
    if not m:
        return None
    amount_formatted = f"${int(m.group(1).replace(',', '')):,}"
    citations = _CITATION_RE.findall(answer)
    citation_str = ' ' + citations[0] if citations else ''
    ticker_m = _TICKER_FROM_CITATION_RE.search(citations[0]) if citations else None
    company = _TICKER_TO_COMPANY.get(ticker_m.group(1), "The company") if ticker_m else "The company"
    return f"{company} reported total net sales of {amount_formatted} million.{citation_str}"


def format_model_answer(answer: str, question: str | None = None) -> str:
    answer = _strip_role_prefix(answer)
    if question and 'total net sales' in question.lower():
        scalar = _extract_total_net_sales(answer)
        if scalar:
            return scalar
    answer = _normalize_whitespace(answer)
    answer = _convert_inline_numbered_list(answer)
    answer = _deduplicate_bullets(answer)
    answer = _drop_truncated_fragments(answer)
    return answer.strip()
