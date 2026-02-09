#!/usr/bin/env python3
"""
Smart Bank Statement Extractor

Hybrid approach that combines agentic and vision extraction:
1. Check if PDF has extractable text
2. If yes → Try agentic (fast, ~18s)
3. If agentic fails or no text → Fall back to vision (~50-180s)

This gives us:
- Fast extraction when agentic works
- Reliable fallback when it doesn't
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pdfplumber

from agentic_extractor import agentic_extract
from vision_extractor import vision_extract

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
    parser.add_argument("--agentic", action="store_true", help="Force agentic only")
    parser.add_argument("--vision", action="store_true", help="Force vision only")

    args = parser.parse_args()

    print(f"=== Smart Extractor ===")
    print(f"PDF: {args.pdf}")
    print()

    start_time = time.time()

    # Handle forced modes
    if args.agentic:
        print("  Using agentic extraction (forced)...")
        transactions, error = agentic_extract(args.pdf)
        result = ExtractionResult(
            pdf_name=Path(args.pdf).name,
            method="agentic",
            transactions=transactions,
            transaction_count=len(transactions),
            time_seconds=round(time.time() - start_time, 2),
            error=error
        )
    elif args.vision:
        print("  Using vision extraction (forced)...")
        transactions, error = vision_extract(args.pdf)
        result = ExtractionResult(
            pdf_name=Path(args.pdf).name,
            method="vision",
            transactions=transactions,
            transaction_count=len(transactions),
            time_seconds=round(time.time() - start_time, 2),
            error=error
        )
    else:
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
