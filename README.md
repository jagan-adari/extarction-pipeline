# Bank Statement Extraction Pipeline

Smart extraction pipeline for bank statements using hybrid agentic + vision approach.

## How It Works

```
PDF Input
    │
    ▼
┌─────────────────────┐
│ Check for text      │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
Has Text?    No Text?
    │           │
    ▼           ▼
┌─────────┐  ┌─────────┐
│ Agentic │  │ Vision  │
│ (~18s)  │  │ (~50s)  │
└────┬────┘  └────┬────┘
     │            │
     ▼            │
Got 3+ txns?      │
     │            │
  ┌──┴──┐         │
  │     │         │
  ▼     ▼         │
 Yes    No ───────┤
  │               │
  ▼               ▼
┌─────────────────────┐
│ Return Transactions │
└─────────────────────┘
```

### Agentic Extraction (~18 seconds)
1. Sample first 3 pages (images + text)
2. Claude analyzes format and writes custom Python parser
3. Execute parser locally
4. Return transactions

### Vision Extraction (~50-180 seconds)
1. Send PDF directly to Claude Vision
2. Claude reads and extracts all transactions
3. Return transactions

## Installation

```bash
pip install pdfplumber anthropic pdf2image pillow
brew install poppler  # Required for pdf2image on macOS
```

## Usage

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key

# Basic extraction
python3 smart_extractor.py statement.pdf

# With verbose output
python3 smart_extractor.py statement.pdf -v

# Save to JSON
python3 smart_extractor.py statement.pdf -o output.json
```

## Output Format

```json
{
  "pdf_name": "statement.pdf",
  "method": "agentic",
  "transactions": [
    {
      "date": "2025-12-04",
      "description": "ACH DEPOSIT PAYROLL",
      "amount": 3500.00,
      "type": "credit"
    },
    {
      "date": "2025-12-05",
      "description": "VISA PURCHASE AMAZON",
      "amount": 125.99,
      "type": "debit"
    }
  ],
  "transaction_count": 169,
  "time_seconds": 17.8
}
```

## Performance

| Method | Time | When Used |
|--------|------|-----------|
| Agentic | ~18s | Text-based PDFs |
| Vision | ~50-180s | Image/scanned PDFs, or agentic fallback |

## Files

```
├── smart_extractor.py   # Main extraction tool
├── keywords.json        # Lender matching keywords
├── data/                # Sample PDFs
└── .env                 # API keys (not in repo)
```

## Lender Matching

Edit `keywords.json` to add lender keywords:

```json
{
  "lenderKeywords": {
    "Fundbox": ["FUNDBOX", "FBX"],
    "OnDeck": ["ONDECK", "ON DECK CAPITAL"]
  },
  "transferKeywords": ["TRANSFER", "ACH", "WIRE"]
}
```

## Environment

Create `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## License

MIT
