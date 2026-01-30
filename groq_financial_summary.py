import csv
import json
import os
import requests
from dotenv import load_dotenv
load_dotenv()
# Load keywords
with open("keywords.json", "r") as f:
    keywords_data = json.load(f)
    lender_keywords = {k: [kw.lower() for kw in v] for k, v in keywords_data.get("businessCategoryKeywords", {}).items()}
    transfer_keywords = [kw.lower() for kw in keywords_data.get("transferKeywords", [])]

# Read transactions
transactions = []
with open("asset_transactions.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        transactions.append(row)

# Prepare prompt for Groq LLM
prompt = f"""
You are a financial analysis assistant. Given the following bank transaction data for a single account, generate a financial summary report in JSON format.

Requirements:
- Only include fields in the JSON if the data is available (do not hardcode or include empty/default fields).
- Do not hardcode any values such as accountId or accountName; infer them from the data if possible, otherwise omit.
- The JSON should have an 'accounts' array, each with summary fields only if data is present.
- Classify each transaction as a transfer, payment, lender payment, or deposit using the following lender and transfer keywords:
Lender keywords: {json.dumps(lender_keywords)}
Transfer keywords: {json.dumps(transfer_keywords)}

Transaction data (CSV):
"""
for tx in transactions:
    # prompt += f"{tx['date']},{tx['amount']},{tx['transaction_type']},{tx['check_number']},{tx['description']}\n"
    prompt += f"{tx['Date']},{tx['Description']},{tx['Inflow']},{tx['Outflow']},{tx['Ending Daily Balance'],}\n"

prompt += "\nGenerate the financial summary result JSON as shown above."

# Call xAI LLM API
XAI_API_KEY = os.environ.get("GROQ_API_KEY")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

if not XAI_API_KEY:
    print("Error: GROQ_API_KEY environment variable not set.")
    exit(1)

headers = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": "grok-3",
    "messages": [
        {"role": "system", "content": "You are a helpful financial analysis assistant."},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 1024,
    "temperature": 0.2
}

response = requests.post(XAI_API_URL, headers=headers, json=data)
if response.status_code == 200:
    result = response.json()
    summary = result["choices"][0]["message"]["content"]
    print("Financial summary result (from xAI LLM):")
    print(summary)
else:
    print(f"xAI API error: {response.status_code}")
    print(response.text)
