import csv
import json
from collections import defaultdict

# Load keywords
with open("keywords.json", "r") as f:
    keywords_data = json.load(f)
    lender_keywords = {k: [kw.lower() for kw in v] for k, v in keywords_data.get("businessCategoryKeywords", {}).items()}
    transfer_keywords = [kw.lower() for kw in keywords_data.get("transferKeywords", [])]

# Helper to match lender
def match_lender(description):
    matches = []
    for lender, kwlist in lender_keywords.items():
        for kw in kwlist:
            if kw and kw in description.lower():
                matches.append(lender)
                break
    return matches

# Helper to match transfer
def is_transfer(description):
    for kw in transfer_keywords:
        if kw and kw in description.lower():
            return True
    return False

# Read transactions
transactions = []
with open("output.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        transactions.append(row)

# Aggregate for single account
account_id = "main"
account_name = "Main Account"
transfers = 0.0
payments = 0.0
lenders = set()
lender_names = set()
deposits = 0.0
aggregate_deposits = 0.0
aggregate_payments = 0.0

for tx in transactions:
    desc = tx["description"] or ""
    amt = float(tx["amount"]) if tx["amount"] else 0.0
    tx_type = tx["transaction_type"].lower()
    lender_match = match_lender(desc)
    transfer = is_transfer(desc)
    # Deposits: all credits
    if tx_type == "credit":
        deposits += amt
        aggregate_deposits += amt
    # Payments: all debits
    elif tx_type == "debit":
        payments += amt
        aggregate_payments += amt
    # Transfers: if transfer keyword
    if transfer:
        transfers += amt
    # Lenders: if lender match
    if lender_match:
        lenders.update(lender_match)
        lender_names.update(lender_match)

summary = (
    f"During October 2025 the {account_name} received ${deposits:,.2f} in inbound credits. "
    f"It made ${payments:,.2f} in debits. "
    f"Total October deposits were ${deposits:,.2f}. "
    f"Lender payments: {', '.join(lender_names) if lender_names else 'None'}. "
    f"Transfers: ${transfers:,.2f}."
)

result = {
    "accounts": [
        {
            "accountId": account_id,
            "accountName": account_name,
            "transfers": round(transfers, 2),
            "payments": round(payments, 2),
            "lenders": len(lenders),
            "lenderNames": list(lender_names),
            "deposits": round(deposits, 2),
            "summary": summary,
            "aggregateDeposits": round(aggregate_deposits, 2),
            "aggregatePayments": round(aggregate_payments, 2)
        }
    ]
}

print("Financial summary result:")
print(json.dumps(result, indent=2))
