from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from finrag.config import (
    DEFAULT_SEC_USER_AGENT,
    DEFAULT_TICKERS,
    RAW_DOCUMENTS_DIR,
    ensure_data_dirs,
)

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
COMPANY_TICKERS_URL = f"{SEC_BASE}/files/company_tickers.json"


def sec_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def get_json(url: str, user_agent: str) -> Any:
    response = requests.get(url, headers=sec_headers(user_agent), timeout=45)
    response.raise_for_status()
    return response.json()


def get_text(url: str, user_agent: str) -> str:
    response = requests.get(url, headers=sec_headers(user_agent), timeout=60)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def ticker_to_cik(user_agent: str) -> dict[str, str]:
    payload = get_json(COMPANY_TICKERS_URL, user_agent)
    return {
        row["ticker"].upper(): str(row["cik_str"]).zfill(10)
        for row in payload.values()
    }


def latest_filing(submissions: dict[str, Any], form: str) -> dict[str, str]:
    recent = submissions["filings"]["recent"]
    for idx, filing_form in enumerate(recent["form"]):
        if filing_form == form:
            return {
                "accession_no": recent["accessionNumber"][idx],
                "filing_date": recent["filingDate"][idx],
                "report_date": recent["reportDate"][idx],
                "primary_document": recent["primaryDocument"][idx],
                "form": filing_form,
            }
    raise ValueError(f"No {form} filing found in recent submissions.")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def download_latest_10k(ticker: str, cik: str, user_agent: str, output_dir: Path) -> Path:
    submissions_url = f"{SEC_DATA_BASE}/submissions/CIK{cik}.json"
    submissions = get_json(submissions_url, user_agent)
    filing = latest_filing(submissions, "10-K")

    accession_path = filing["accession_no"].replace("-", "")
    cik_int = str(int(cik))
    doc_url = (
        f"{SEC_BASE}/Archives/edgar/data/{cik_int}/"
        f"{accession_path}/{filing['primary_document']}"
    )

    raw_html = get_text(doc_url, user_agent)
    text = html_to_text(raw_html)

    company = submissions.get("name", ticker)
    doc_id = f"{ticker}_{filing['filing_date']}_{filing['form']}"
    filename_base = safe_name(doc_id)
    text_path = output_dir / f"{filename_base}.txt"
    metadata_path = output_dir / f"{filename_base}.json"

    text_path.write_text(text, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "ticker": ticker,
                "company": company,
                "cik": cik,
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "report_date": filing["report_date"],
                "accession_no": filing["accession_no"],
                "primary_document": filing["primary_document"],
                "source_url": doc_url,
                "source": f"{company} {filing['form']} filed {filing['filing_date']}",
                "text_file": text_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return text_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download latest SEC 10-K filings.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to download.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_SEC_USER_AGENT,
        help="SEC User-Agent header. Set SEC_USER_AGENT in .env for your contact info.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DOCUMENTS_DIR,
        help="Directory for extracted filing text and metadata.",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between SEC calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = ticker_to_cik(args.user_agent)
    downloaded: list[Path] = []

    for ticker in tqdm([ticker.upper() for ticker in args.tickers], desc="SEC filings"):
        if ticker not in mapping:
            raise KeyError(f"Ticker not found in SEC company_tickers.json: {ticker}")
        downloaded.append(download_latest_10k(ticker, mapping[ticker], args.user_agent, args.output_dir))
        time.sleep(args.sleep)

    print("Downloaded filings:")
    for path in downloaded:
        print(f"- {path}")


if __name__ == "__main__":
    main()
