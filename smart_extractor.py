#!/usr/bin/env python3
"""
Smart Bank Statement Extractor

Strategy:
1. Check if PDF has extractable text
2. If yes → Try agentic parser (Claude writes custom code)
3. If agentic fails or no text → Fall back to vision

This gives us:
- Fast, cheap extraction when agentic works (~5s, 1 API call)
- Reliable fallback via vision when it doesn't (~50s, more expensive)
"""

import os
import json
import time
import base64
import subprocess
import tempfile
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass, asdict
from typing import Optional

import pdfplumber
import anthropic
from pdf2image import convert_from_path

MODEL = "claude-sonnet-4-20250514"
client = anthropic.Anthropic()

# Minimum text per page to consider text-extractable
MIN_TEXT_PER_PAGE = 100

# Minimum transactions to consider extraction successful
MIN_TRANSACTIONS = 3


@dataclass
class ExtractionResult:
    pdf_name: str
    method: str  # "agentic" or "vision"
    transactions: list
    transaction_count: int
    time_seconds: float
    error: Optional[str] = None


# ============================================================
# AGENTIC EXTRACTION
# ============================================================

def agentic_extract(pdf_path: str) -> tuple[list, str]:
    """
    Have Claude analyze the PDF and write a custom parser.
    Returns: (transactions, error_message)
    """
    pdf_path = Path(pdf_path)

    # Get sample text and images
    with pdfplumber.open(pdf_path) as pdf:
        pages_to_sample = min(3, len(pdf.pages))
        text = ""
        for i in range(pages_to_sample):
            page_text = pdf.pages[i].extract_text() or ""
            text += f"\n=== PAGE {i+1} ===\n{page_text}"

    images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=pages_to_sample)

    # Build content for Claude
    content = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}
        })

    prompt = f"""Write a Python script to extract ALL transactions from this bank statement.

EXTRACTED TEXT:
{text[:12000]}

Requirements:
1. Use pdfplumber to open: "{pdf_path}"
2. Extract ALL transactions from ALL pages
3. Handle the specific format in this statement (sections, dates, amounts)
4. Output JSON array to stdout

Output format:
[{{"date": "YYYY-MM-DD", "description": "...", "amount": 123.45, "type": "credit" or "debit"}}]

Important:
- Identify ALL sections (deposits=credit, withdrawals/payments=debit)
- Handle the date format used in this statement
- Handle multi-line descriptions if present
- Skip headers, subtotals, page numbers
- Print ONLY the JSON array

Return ONLY Python code starting with ```python"""

    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}]
    )

    response_text = response.content[0].text

    # Extract code
    if "```python" in response_text:
        code = response_text.split("```python")[1].split("```")[0]
    elif "```" in response_text:
        code = response_text.split("```")[1].split("```")[0]
    else:
        return [], "No code in response"

    # Execute code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        parser_path = f.name

    try:
        result = subprocess.run(
            ["python3", parser_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        os.unlink(parser_path)

        if result.returncode != 0:
            return [], f"Script error: {result.stderr[:500]}"

        output = result.stdout.strip()
        if "[" in output:
            json_str = output[output.index("["):output.rindex("]")+1]
            transactions = json.loads(json_str)
            return transactions, None
        return [], "No JSON in output"

    except subprocess.TimeoutExpired:
        os.unlink(parser_path)
        return [], "Script timeout"
    except Exception as e:
        return [], str(e)


# ============================================================
# VISION EXTRACTION
# ============================================================

def vision_extract(pdf_path: str) -> tuple[list, str]:
    """
    Use Claude Vision to directly extract transactions.
    Returns: (transactions, error_message)
    """
    pdf_path = Path(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

    # For large PDFs, use page-wise extraction
    if page_count > 10:
        return vision_extract_pagewise(pdf_path, page_count)

    # For small PDFs, send whole document
    with open(pdf_path, "rb") as f:
        pdf_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model=MODEL,
        max_tokens=16384,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}
                },
                {
                    "type": "text",
                    "text": """Extract ALL transactions from this bank statement.

Return JSON array:
[{"date": "YYYY-MM-DD", "description": "...", "amount": 123.45, "type": "credit" or "debit"}]

Include EVERY transaction. Return ONLY the JSON array."""
                }
            ]
        }]
    )

    return parse_vision_response(response.content[0].text)


def vision_extract_pagewise(pdf_path: Path, page_count: int) -> tuple[list, str]:
    """Extract from large PDFs page by page."""
    pages = convert_from_path(pdf_path, dpi=150)
    all_transactions = []

    batch_size = 5
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]

        content = []
        for page in batch:
            buffered = BytesIO()
            page.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}
            })

        content.append({
            "type": "text",
            "text": "Extract ALL transactions from these bank statement pages. Return JSON array: [{\"date\": \"YYYY-MM-DD\", \"description\": \"...\", \"amount\": 123.45, \"type\": \"credit\" or \"debit\"}]. Return ONLY JSON."
        })

        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": content}]
        )

        batch_txns, _ = parse_vision_response(response.content[0].text)
        all_transactions.extend(batch_txns)

    # Deduplicate
    seen = set()
    unique = []
    for t in all_transactions:
        key = (t.get("date"), t.get("description", "")[:30], t.get("amount"))
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique, None


def parse_vision_response(text: str) -> tuple[list, str]:
    """Parse Claude's response to extract JSON."""
    try:
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0]
        elif "[" in text:
            json_str = text[text.index("["):text.rindex("]")+1]
        else:
            return [], "No JSON found"

        return json.loads(json_str.strip()), None
    except Exception as e:
        return [], str(e)


# ============================================================
# SMART EXTRACTION (MAIN)
# ============================================================

def smart_extract(pdf_path: str) -> ExtractionResult:
    """
    Smart extraction: tries agentic first, falls back to vision.
    """
    start_time = time.time()
    pdf_path = Path(pdf_path)

    # Check if PDF has extractable text
    with pdfplumber.open(pdf_path) as pdf:
        total_chars = sum(len(p.extract_text() or "") for p in pdf.pages)
        avg_chars = total_chars / len(pdf.pages) if pdf.pages else 0

    has_text = avg_chars >= MIN_TEXT_PER_PAGE

    # Try agentic first if there's text
    if has_text:
        print(f"  Trying agentic extraction...")
        transactions, error = agentic_extract(str(pdf_path))

        if not error and len(transactions) >= MIN_TRANSACTIONS:
            return ExtractionResult(
                pdf_name=pdf_path.name,
                method="agentic",
                transactions=transactions,
                transaction_count=len(transactions),
                time_seconds=round(time.time() - start_time, 2)
            )

        print(f"  Agentic failed ({error or 'too few transactions'}), falling back to vision...")

    # Fall back to vision
    print(f"  Using vision extraction...")
    transactions, error = vision_extract(str(pdf_path))

    return ExtractionResult(
        pdf_name=pdf_path.name,
        method="vision",
        transactions=transactions,
        transaction_count=len(transactions),
        time_seconds=round(time.time() - start_time, 2),
        error=error
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smart Bank Statement Extractor")
    parser.add_argument("pdf", help="PDF file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    print(f"=== Smart Extractor ===")
    print(f"PDF: {args.pdf}")
    print()

    result = smart_extract(args.pdf)

    print(f"\n=== Results ===")
    print(f"Method: {result.method}")
    print(f"Transactions: {result.transaction_count}")
    print(f"Time: {result.time_seconds}s")

    if result.error:
        print(f"Error: {result.error}")

    if args.verbose and result.transactions:
        print(f"\nSample transactions:")
        for t in result.transactions[:5]:
            print(f"  {t.get('date')} | {t.get('description', '')[:40]:<40} | {t.get('amount', 0):>10} | {t.get('type')}")

    if args.output:
        output = {
            "pdf_name": result.pdf_name,
            "method": result.method,
            "transactions": result.transactions,
            "transaction_count": result.transaction_count,
            "time_seconds": result.time_seconds,
            "error": result.error
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
