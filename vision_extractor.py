#!/usr/bin/env python3
"""
Vision Extractor: Claude reads PDF directly with vision.

How it works:
1. For small PDFs (≤10 pages): Send entire PDF to Claude
2. For large PDFs (>10 pages): Process in batches of 5 pages
3. Claude extracts transactions directly from visual content
4. Return extracted transactions

Speed: ~50-180 seconds
Best for: Scanned/image PDFs, complex layouts, agentic fallback
"""

import json
import time
import base64
from pathlib import Path
from io import BytesIO

import pdfplumber
import anthropic
from pdf2image import convert_from_path

MODEL = "claude-sonnet-4-20250514"
client = anthropic.Anthropic()


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
    """Extract from large PDFs page by page in batches."""
    pages = convert_from_path(pdf_path, dpi=150)
    all_transactions = []

    batch_size = 5
    total_batches = (len(pages) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(pages), batch_size)):
        batch = pages[i:i + batch_size]
        print(f"  Processing batch {batch_num + 1}/{total_batches}...")

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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Vision Bank Statement Extractor")
    parser.add_argument("pdf", help="PDF file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    print(f"=== Vision Extractor ===")
    print(f"PDF: {args.pdf}")

    start_time = time.time()
    transactions, error = vision_extract(args.pdf)
    elapsed = round(time.time() - start_time, 2)

    print(f"\nTransactions: {len(transactions)}")
    print(f"Time: {elapsed}s")

    if error:
        print(f"Error: {error}")

    if args.verbose and transactions:
        print(f"\nSample:")
        for t in transactions[:5]:
            print(f"  {t.get('date')} | {t.get('description', '')[:40]:<40} | {t.get('amount', 0):>10} | {t.get('type')}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"transactions": transactions, "count": len(transactions), "time": elapsed}, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
