import json
import csv
import re
from docling.document_converter import DocumentConverter

source = "/Users/jaganadari/Documents/Projects/Loop/10 (1).pdf"
converter = DocumentConverter()
doc = converter.convert(source).document

# Load keywords.json
with open("keywords.json", "r") as f:
	keywords_data = json.load(f)
	lender_keywords = keywords_data.get("businessCategoryKeywords", {})
	transfer_keywords = [kw.lower() for kw in keywords_data.get("transferKeywords", [])]

# Extract table from markdown (assumes doc.export_to_markdown() is a table)

markdown = doc.export_to_markdown()
# Write markdown output to a file for inspection
with open("output.md", "w") as mdfile:
	mdfile.write(markdown)
print("Markdown output written to output.md for inspection.")

lines = markdown.splitlines()

# --- Checks table extraction logic ---
headers = ["date", "amount", "transaction_type", "check_number", "lender_matches", "is_lender_transfer", "is_lender_payment", "description"]
output_rows = []

def add_row(date, amount, transaction_type, check_number, description):
    # Only add if at least date and amount or description and amount are present
    if (date or description) and amount:
        output_rows.append({
            "date": date,
            "amount": amount,
            "transaction_type": transaction_type,
            "check_number": check_number,
            "lender_matches": "",
            "is_lender_transfer": False,
            "is_lender_payment": False,
            "description": description
        })

# --- Improved extraction logic for multi-line and split table rows ---
i = 0
current_table_type = None
pending_row = None
while i < len(lines):
    line = lines[i]
    # Detect table headers
    if line.strip().startswith('| DATE') and 'CHECK #' in line and 'AMOUNT' in line:
        current_table_type = 'checks'
        i += 2
        continue
    elif line.strip().startswith('| DATE') and 'DESCRIPTION' in line and 'AMOUNT' in line:
        current_table_type = 'desc_amount'
        i += 2
        continue
    elif line.strip().startswith('|') and set(line.strip()) == {'|', '-'}:
        # Table separator, skip
        i += 1
        continue
    # Parse table rows
    if current_table_type == 'checks' and line.strip().startswith('|'):
        cols = [c.strip() for c in line.strip('|').split('|')]
        if len(cols) >= 3:
            date = cols[0]
            check_number = cols[1].replace('*', '').strip()
            amount = cols[2].replace('$', '').replace(',', '').strip()
            # Skip summary/total rows
            if date.lower() == 'total' or not amount or amount.lower() == 'amount($)':
                i += 1
                continue
            try:
                float(amount)
            except ValueError:
                i += 1
                continue
            add_row(date, amount, "debit", check_number, "Check")
        i += 1
        continue
    elif current_table_type == 'desc_amount' and line.strip().startswith('|'):
        cols = [c.strip() for c in line.strip('|').split('|')]
        # Handle multi-line/split rows
        if len(cols) == 3:
            date, description, amount = cols
            if not date and pending_row:
                # This is a continuation of previous row's description/amount
                pending_row['description'] += ' ' + description
                if amount:
                    pending_row['amount'] = amount.replace('$', '').replace(',', '').strip()
                    try:
                        amt = float(pending_row['amount'])
                    except ValueError:
                        i += 1
                        continue
                    transaction_type = "credit" if amt > 0 else "debit"
                    add_row(pending_row['date'], pending_row['amount'], transaction_type, "", pending_row['description'])
                    pending_row = None
            else:
                # New row
                if amount:
                    try:
                        amt = float(amount.replace('$', '').replace(',', '').strip())
                    except ValueError:
                        i += 1
                        continue
                    transaction_type = "credit" if amt > 0 else "debit"
                    add_row(date, amount.replace('$', '').replace(',', '').strip(), transaction_type, "", description)
                    pending_row = None
                else:
                    # Start pending row for multi-line
                    pending_row = {'date': date, 'description': description, 'amount': ''}
        elif len(cols) == 2 and pending_row:
            # Sometimes amount is on a separate line
            description, amount = cols
            pending_row['description'] += ' ' + description
            if amount:
                pending_row['amount'] = amount.replace('$', '').replace(',', '').strip()
                try:
                    amt = float(pending_row['amount'])
                except ValueError:
                    i += 1
                    continue
                transaction_type = "credit" if amt > 0 else "debit"
                add_row(pending_row['date'], pending_row['amount'], transaction_type, "", pending_row['description'])
                pending_row = None
        else:
            # Not a valid row, reset pending
            pending_row = None
        i += 1
        continue
    else:
        current_table_type = None
        pending_row = None
        i += 1
# --- End improved extraction logic ---
# --- End extraction logic ---

# Write to CSV
with open("output.csv", "w", newline="") as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=headers)
	writer.writeheader()
	for row in output_rows:
		writer.writerow(row)

print(f"Extracted {len(output_rows)} transactions to output.csv")