#!/usr/bin/env python3
"""
Agentic Extractor: Claude analyzes PDF and writes a custom parser.

How it works:
1. Sample first 3 pages (images + text)
2. Send to Claude with prompt to write a parser
3. Claude returns Python code
4. Execute the code locally
5. Return extracted transactions

Speed: ~18 seconds
Best for: Text-based PDFs with consistent formats
"""

import os
import json
import time
import base64
import subprocess
import tempfile
from pathlib import Path
from io import BytesIO

import pdfplumber
import anthropic
from pdf2image import convert_from_path

MODEL = "claude-sonnet-4-20250514"
client = anthropic.Anthropic()


def agentic_extract(pdf_path: str) -> tuple[list, str]:
    """
    Have Claude analyze the PDF and write a custom parser.
    Returns: (transactions, error_message)
    """
    pdf_path = Path(pdf_path)

    # Get sample text and images from first 3 pages
    with pdfplumber.open(pdf_path) as pdf:
        pages_to_sample = min(3, len(pdf.pages))
        text = ""
        for i in range(pages_to_sample):
            page_text = pdf.pages[i].extract_text() or ""
            text += f"\n=== PAGE {i+1} ===\n{page_text}"

    images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=pages_to_sample)

    # Build content for Claude (images + prompt)
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

    # Call Claude to generate parser code
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}]
    )

    response_text = response.content[0].text

    # Extract code from response
    if "```python" in response_text:
        code = response_text.split("```python")[1].split("```")[0]
    elif "```" in response_text:
        code = response_text.split("```")[1].split("```")[0]
    else:
        return [], "No code in response"

    # Execute the generated code
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agentic Bank Statement Extractor")
    parser.add_argument("pdf", help="PDF file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    print(f"=== Agentic Extractor ===")
    print(f"PDF: {args.pdf}")

    start_time = time.time()
    transactions, error = agentic_extract(args.pdf)
    elapsed = round(time.time() - start_time, 2)

    print(f"\nTransactions: {len(transactions)}")
    print(f"Time: {elapsed}s")

    if error:
        print(f"Error: {error}")

    if args.verbose and transactions:
        print(f"\nSample:")
        for t in transactions[:5]:
            print(f"  {t['date']} | {t['description'][:40]:<40} | {t['amount']:>10} | {t['type']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"transactions": transactions, "count": len(transactions), "time": elapsed}, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
