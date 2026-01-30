#!/usr/bin/env python3
"""
Extract data from Plaid Asset Report PDF
"""

import re
import json
import csv
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import subprocess
    subprocess.check_call(["pip", "install", "pdfplumber"])
    import pdfplumber


@dataclass
class ReportInfo:
    requester_report_id: str = ""
    requester_user_id: str = ""
    requested_on: str = ""
    days_requested: int = 0
    accounts: int = 0


@dataclass
class BorrowerInfo:
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    ssn: str = ""
    phone_number: str = ""
    email: str = ""


@dataclass
class AccountSummary:
    institution: str = ""
    account_name: str = ""
    account_mask: str = ""
    current_balance: float = 0.0
    available_balance: float = 0.0
    account_type: str = ""


@dataclass
class Transaction:
    date: str = ""
    description: str = ""
    inflow: Optional[float] = None
    outflow: Optional[float] = None
    ending_daily_balance: Optional[float] = None
    is_pending: bool = False


@dataclass
class AccountDetail:
    institution: str = ""
    account_name: str = ""
    account_mask: str = ""
    holder_name: str = ""
    address: str = ""
    phone: str = ""
    email: str = ""
    current_balance: float = 0.0
    available_balance: float = 0.0
    account_type: str = ""
    transactions: list = field(default_factory=list)


class AssetReportExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.report_info = ReportInfo()
        self.borrower_info = BorrowerInfo()
        self.asset_summary: list[AccountSummary] = []
        self.account_details: list[AccountDetail] = []
        self.pages_data = []  # List of (page_num, text, account_context)

    def extract(self) -> dict:
        """Extract all data from the PDF."""
        with pdfplumber.open(self.pdf_path) as pdf:
            current_account = None

            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                # Detect account context from page header
                if "ACCOUNT MASK: 7570" in text or "Payroll Account" in text:
                    if "Account Overview" in text or "ACCOUNT NAME: Payroll" in text:
                        current_account = "7570"
                elif "ACCOUNT MASK: 7562" in text or "General Expense" in text:
                    if "Account Overview" in text or "ACCOUNT NAME: General" in text:
                        current_account = "7562"

                self.pages_data.append({
                    'page_num': page_num + 1,
                    'text': text,
                    'account': current_account
                })

        self.full_text = "\n".join([p['text'] for p in self.pages_data])

        self._extract_report_info()
        self._extract_borrower_info()
        self._extract_asset_summary()
        self._extract_account_details_with_transactions()

        return self.to_dict()

    def _extract_report_info(self):
        """Extract report information section."""
        patterns = {
            'requester_report_id': r'Requester report ID\s+(\d+)',
            'requester_user_id': r'Requester user ID\s+(\d+)',
            'requested_on': r'Requested on\s+([A-Za-z]+ \d+, \d+)',
            'days_requested': r'Days requested\s+(\d+)',
            'accounts': r'Accounts\s+(\d+)',
        }

        for field_name, pattern in patterns.items():
            match = re.search(pattern, self.full_text)
            if match:
                value = match.group(1)
                if field_name in ['days_requested', 'accounts']:
                    value = int(value)
                setattr(self.report_info, field_name, value)

    def _extract_borrower_info(self):
        """Extract borrower information section."""
        patterns = {
            'first_name': r'First name\s+([A-Za-z]+)',
            'last_name': r'Last name\s+([A-Za-z]+)',
            'phone_number': r'Phone number\s+(\d+)',
            'email': r'Email\s+([^\s]+@[^\s]+)',
        }

        for field_name, pattern in patterns.items():
            match = re.search(pattern, self.full_text)
            if match:
                setattr(self.borrower_info, field_name, match.group(1))

    def _extract_asset_summary(self):
        """Extract asset summary section."""
        account_pattern = r'(Centra Credit Union)\s+([\w\s]+?)\s+(\d{4})\s+\$([0-9,]+\.\d{2})'
        matches = re.findall(account_pattern, self.full_text[:3000])

        for match in matches:
            account = AccountSummary(
                institution=match[0].strip(),
                account_name=match[1].strip(),
                account_mask=match[2],
                current_balance=self._parse_amount(match[3])
            )
            if not any(a.account_mask == account.account_mask for a in self.asset_summary):
                self.asset_summary.append(account)

    def _extract_account_details_with_transactions(self):
        """Extract account details and transactions using page-based approach."""
        accounts_config = [
            {"name": "Payroll Account", "mask": "7570", "start_page": 2, "end_page": 6},
            {"name": "General Expense", "mask": "7562", "start_page": 7, "end_page": 36},
        ]

        for acc_config in accounts_config:
            account = AccountDetail(
                institution="Centra Credit Union",
                account_name=acc_config["name"],
                account_mask=acc_config["mask"],
                account_type="Depository",
                holder_name=f"{self.borrower_info.first_name} {self.borrower_info.last_name}"
            )

            # Extract balances from the account summary we already have
            for summary in self.asset_summary:
                if summary.account_mask == acc_config["mask"]:
                    account.current_balance = summary.current_balance
                    break

            # Extract transactions from relevant pages
            transactions = []
            for page_data in self.pages_data:
                page_num = page_data['page_num']

                # Only process pages for this account
                if page_num < acc_config["start_page"] or page_num > acc_config["end_page"]:
                    continue

                page_transactions = self._extract_transactions_from_text(page_data['text'])
                transactions.extend(page_transactions)

            account.transactions = transactions
            self.account_details.append(account)

    def _extract_transactions_from_text(self, text: str) -> list[dict]:
        """Extract transactions from page text."""
        transactions = []
        lines = text.split('\n')

        # Date pattern for transaction lines
        date_pattern = r'^((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})'

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip non-transaction lines
            if not line or 'PAGE' in line or 'Account History' in line:
                i += 1
                continue
            if line.startswith('Date') or line == 'Posted' or line == 'Pending':
                i += 1
                continue

            # Check if line starts with a date
            date_match = re.match(date_pattern, line)
            if date_match:
                trans = self._parse_transaction_line(line)
                if trans:
                    transactions.append(trans)

            i += 1

        return transactions

    def _parse_transaction_line(self, line: str) -> Optional[dict]:
        """Parse a single transaction line."""
        # Pattern: Date Description Amount1 Amount2 [Amount3]
        # Amounts can be $X.XX or ---

        date_pattern = r'^((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\s+'
        date_match = re.match(date_pattern, line)

        if not date_match:
            return None

        date = date_match.group(1)
        rest = line[date_match.end():].strip()

        # Find amounts at the end (they look like $1,234.56 or ---)
        amount_pattern = r'(\$[0-9,]+\.\d{2}|---)'
        amounts = re.findall(amount_pattern, rest)

        if len(amounts) < 2:
            return None

        # Remove amounts from the end to get description
        description = rest
        for amt in amounts:
            # Remove from right side
            idx = description.rfind(amt)
            if idx != -1:
                description = description[:idx].strip()

        # Parse amounts
        inflow_str = amounts[0] if len(amounts) > 0 else '---'
        outflow_str = amounts[1] if len(amounts) > 1 else '---'
        balance_str = amounts[2] if len(amounts) > 2 else '---'

        return {
            'date': date,
            'description': description.strip(),
            'inflow': self._parse_amount(inflow_str) if inflow_str != '---' else None,
            'outflow': self._parse_amount(outflow_str) if outflow_str != '---' else None,
            'ending_daily_balance': self._parse_amount(balance_str) if balance_str != '---' else None,
            'is_pending': False
        }

    def _parse_amount(self, amount_str: str) -> float:
        """Parse currency string to float."""
        if not amount_str or amount_str == '---':
            return 0.0
        cleaned = amount_str.replace('$', '').replace(',', '').strip()
        if not cleaned:
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def to_dict(self) -> dict:
        """Convert all extracted data to dictionary."""
        return {
            'report_info': asdict(self.report_info),
            'borrower_info': asdict(self.borrower_info),
            'asset_summary': [asdict(a) for a in self.asset_summary],
            'account_details': [
                {
                    'institution': a.institution,
                    'account_name': a.account_name,
                    'account_mask': a.account_mask,
                    'holder_name': a.holder_name,
                    'current_balance': a.current_balance,
                    'available_balance': a.available_balance,
                    'account_type': a.account_type,
                    'transactions': a.transactions,
                    'transaction_count': len(a.transactions)
                }
                for a in self.account_details
            ],
            'total_balance': sum(a.current_balance for a in self.asset_summary)
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert extracted data to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self):
        """Print a human-readable summary."""
        data = self.to_dict()

        print("=" * 60)
        print("ASSET REPORT SUMMARY")
        print("=" * 60)

        print("\n📋 REPORT INFORMATION")
        print(f"  Report ID: {data['report_info']['requester_report_id']}")
        print(f"  Requested: {data['report_info']['requested_on']}")
        print(f"  Days: {data['report_info']['days_requested']}")

        print("\n👤 BORROWER INFORMATION")
        bi = data['borrower_info']
        print(f"  Name: {bi['first_name']} {bi['last_name']}")
        print(f"  Phone: {bi['phone_number']}")
        print(f"  Email: {bi['email']}")

        print("\n💰 ASSET SUMMARY")
        print(f"  Total Balance: ${data['total_balance']:,.2f}")
        print("\n  Accounts:")
        for acc in data['asset_summary']:
            print(f"    - {acc['account_name']} (****{acc['account_mask']}): ${acc['current_balance']:,.2f}")

        print("\n📊 ACCOUNT DETAILS")
        total_inflow = 0
        total_outflow = 0

        for acc in data['account_details']:
            print(f"\n  {acc['institution']} - {acc['account_name']}")
            print(f"    Mask: ****{acc['account_mask']}")
            print(f"    Current Balance: ${acc['current_balance']:,.2f}")
            print(f"    Transaction Count: {acc['transaction_count']}")

            # Calculate totals for this account
            acc_inflow = sum(t['inflow'] or 0 for t in acc['transactions'])
            acc_outflow = sum(t['outflow'] or 0 for t in acc['transactions'])
            total_inflow += acc_inflow
            total_outflow += acc_outflow

            print(f"    Total Inflow: ${acc_inflow:,.2f}")
            print(f"    Total Outflow: ${acc_outflow:,.2f}")

            # Show first few transactions
            if acc['transactions']:
                print(f"    Recent Transactions:")
                for trans in acc['transactions'][:3]:
                    inflow = f"${trans['inflow']:,.2f}" if trans['inflow'] else "---"
                    outflow = f"${trans['outflow']:,.2f}" if trans['outflow'] else "---"
                    print(f"      {trans['date']}: {trans['description'][:40]} | In: {inflow} | Out: {outflow}")

        print(f"\n📈 TOTALS ACROSS ALL ACCOUNTS")
        print(f"  Total Inflow: ${total_inflow:,.2f}")
        print(f"  Total Outflow: ${total_outflow:,.2f}")
        print("\n" + "=" * 60)


def main():
    pdf_path = Path(__file__).parent / "asset_report.pdf"

    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return

    print(f"Extracting data from: {pdf_path}")
    print()

    extractor = AssetReportExtractor(str(pdf_path))
    data = extractor.extract()

    # Print summary
    extractor.print_summary()

    # Save to JSON
    output_path = pdf_path.with_suffix('.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Full data saved to: {output_path}")

    # Save all transactions to CSV
    csv_path = pdf_path.parent / "asset_transactions.csv"
    all_transactions = []

    for acc in data['account_details']:
        for trans in acc['transactions']:
            all_transactions.append({
                'account': acc['account_name'],
                'account_mask': acc['account_mask'],
                **trans
            })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Description', 'Inflow', 'Outflow', 'Ending Daily Balance'])

        for trans in all_transactions:
            # Format amounts properly
            inflow = f"${trans['inflow']:,.2f}" if trans.get('inflow') else '---'
            outflow = f"${trans['outflow']:,.2f}" if trans.get('outflow') else '---'
            balance = f"${trans['ending_daily_balance']:,.2f}" if trans.get('ending_daily_balance') else '---'

            writer.writerow([
                trans.get('date', ''),
                trans.get('description', ''),
                inflow,
                outflow,
                balance
            ])

    print(f"Transactions saved to: {csv_path}")
    print(f"   Total transactions: {len(all_transactions)}")


if __name__ == "__main__":
    main()
