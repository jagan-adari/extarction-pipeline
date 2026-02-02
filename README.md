# PDF Extraction Pipeline

Fast, deterministic extraction pipeline for loan underwriting data from PDFs.

## Features

- **Multi-format support**: Plaid asset reports, Truist statements, generic bank statements
- **Multi-account detection**: Handles PDFs with multiple accounts
- **Lender matching**: Exact + fuzzy keyword matching (rapidfuzz)
- **Underwriting metrics**: Debt service ratio, parse confidence
- **Optional LLM summaries**: xAI/Grok integration for natural language summaries
- **Deterministic**: No LLM in hot path, consistent reproducible results

## Installation

```bash
pip install pdfplumber rapidfuzz python-dotenv openai
```

## Usage

```bash
# Basic extraction
python fast_extract.py "statement.pdf" -o output.json

# With LLM summaries
python fast_extract.py "statement.pdf" --llm -o output.json

# Custom keywords file
python fast_extract.py "statement.pdf" -k custom_keywords.json -o output.json
```

## Output

```json
{
  "extraction_date": "2026-01-31",
  "source_file": "statement.pdf",
  "accounts": [
    {
      "accountId": "1234567890",
      "accountName": "Business Checking",
      "deposits": 50000.00,
      "internalTransfers": 10000.00,
      "lenderCredits": 5000.00,
      "payments": 3000.00,
      "lenders": 2,
      "lenderNames": ["Fundbox", "Intuit Financing"],
      "debtServiceRatio": 0.06,
      "parseConfidence": 1.0,
      "aggregateDeposits": 150000.00,
      "aggregatePayments": 9000.00,
      "summary": "Monthly deposits: $50,000.00..."
    }
  ]
}
```

## Keywords Configuration

Edit `keywords.json` to add/modify lender and transfer keywords:

```json
{
  "lenderKeywords": {
    "Fundbox": ["FUNDBOX", "FBX"],
    "Intuit Financing": ["INTUIT FIN", "QBO CAPITAL"]
  },
  "transferKeywords": ["TRANSFER", "ACH", "WIRE", "XFER"]
}
```

## Environment Variables

For LLM summaries, create a `.env` file:

```
XAI_API_KEY=your_xai_api_key
```

## License

MIT
