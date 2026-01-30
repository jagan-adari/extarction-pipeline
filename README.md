# Financial Data Extraction Pipeline

A Python toolkit for extracting and analyzing financial data from PDF documents. This project provides tools to parse bank statements, Plaid Asset Reports, and generate AI-powered financial summaries.

## Features

- **PDF Transaction Extraction**: Extract transactions from bank statement PDFs using Docling
- **Plaid Asset Report Parser**: Extract structured data from Plaid Asset Report PDFs
- **AI-Powered Summaries**: Generate financial summaries using LLM APIs (xAI/Grok)
- **Transaction Classification**: Automatically classify transactions using keyword matching

## Scripts

### `extract_pdf.py`
Extracts transaction data from bank statement PDFs and exports to CSV/Markdown format.

### `extract_asset.py`
Parses Plaid Asset Report PDFs to extract:
- Report metadata
- Borrower information
- Account summaries
- Transaction history

Outputs data as JSON and CSV files.

### `groq_financial_summary.py`
Uses AI to analyze transaction data and generate financial summary reports with transaction classification.

## Requirements

- Python 3.8+
- pdfplumber
- docling
- python-dotenv
- requests

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jagan-adari/extarction-pipeline.git
   cd extarction-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install pdfplumber docling python-dotenv requests
   ```

3. Create your environment file:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and add your API key:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```

## Configuration - Using Your Own Files

### For `extract_pdf.py`

Update the `source` variable on **line 6** with your PDF path:

```python
source = "/path/to/your/bank_statement.pdf"
```

The script expects a bank statement PDF with tables containing:
- DATE, CHECK #, AMOUNT columns (for check transactions)
- DATE, DESCRIPTION, AMOUNT columns (for other transactions)

### For `extract_asset.py`

Update the `pdf_path` in the `main()` function on **line 372**:

```python
pdf_path = Path("/path/to/your/asset_report.pdf")
```

You may also need to adjust the `accounts_config` on **lines 165-168** to match your account structure:

```python
accounts_config = [
    {"name": "Your Account Name", "mask": "1234", "start_page": 2, "end_page": 10},
    # Add more accounts as needed
]
```

### For `groq_financial_summary.py`

This script reads from `asset_transactions.csv` by default. If your CSV file has a different name, update **line 15**:

```python
with open("your_transactions.csv", newline="") as csvfile:
```

### Keywords Configuration

Edit `keywords.json` to customize transaction classification:

```json
{
  "businessCategoryKeywords": {
    "category_name": ["keyword1", "keyword2"]
  },
  "transferKeywords": ["transfer", "wire", "ach"]
}
```

## Usage

```bash
# Extract data from Plaid Asset Report
python extract_asset.py

# Extract transactions from bank statement PDF
python extract_pdf.py

# Generate AI financial summary
python groq_financial_summary.py
```

## Output Files

- `asset_report.json` - Structured asset report data
- `asset_transactions.csv` - Transaction history in CSV format
- `output.csv` - Extracted bank statement transactions
- `output.md` - Markdown formatted output

## License

MIT
