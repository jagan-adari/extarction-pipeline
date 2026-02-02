#!/usr/bin/env python3
"""
Fast PDF extraction pipeline - Standalone version

Supports:
- Plaid asset reports (multi-account)
- Bank statements (Truist, generic)

Usage:
  python fast_extract.py "asset_report.pdf" -o output.json
  python fast_extract.py "statement.pdf" --llm -o output.json
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime, date, timedelta
from calendar import monthrange
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pdfplumber

# Optional fuzzy matching
try:
    from rapidfuzz import fuzz, process
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False


# ===============================
# Models
# ===============================
class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class TransactionCategory(str, Enum):
    LENDER_CREDIT = "lender_credit"
    LENDER_PAYMENT = "lender_payment"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    OTHER = "other"


@dataclass
class Transaction:
    date: date
    description: str
    amount: float
    type: TransactionType
    category: Optional[TransactionCategory] = None
    matched_lender: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Account:
    account_id: str
    account_name: str
    transactions: List[Transaction] = field(default_factory=list)
    parse_confidence: float = 1.0


@dataclass
class AccountSummary:
    account_id: str
    account_name: str
    monthly_transfers_in: float = 0.0
    monthly_internal_transfers: float = 0.0
    monthly_lender_credits: float = 0.0
    monthly_lender_payments: float = 0.0
    monthly_deposits: float = 0.0
    monthly_unique_lenders: int = 0
    monthly_lender_names: List[str] = field(default_factory=list)
    aggregate_deposits: float = 0.0
    aggregate_lender_payments: float = 0.0
    aggregate_lender_credits: float = 0.0
    debt_service_ratio: float = 0.0
    parse_confidence: float = 1.0
    summary: str = ""


# ===============================
# Smart Parser
# ===============================
class SmartParser:
    """Parses PDFs - handles Plaid reports and bank statements."""

    def __init__(self):
        self._year = None
        self._doc_type = None

    def parse_pdf(self, pdf_path: Path) -> List[Account]:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        self._doc_type = self._detect_doc_type(full_text)
        self._year = self._detect_year(full_text)

        account_sections = self._find_accounts(full_text)

        if not account_sections:
            account_sections = [("primary", "Primary", full_text)]

        accounts = []
        for acc_id, acc_name, section_text in account_sections:
            transactions = self._parse_transactions(section_text)
            if transactions:
                confidence = self._calculate_confidence(acc_id, acc_name, transactions)
                accounts.append(Account(
                    account_id=acc_id,
                    account_name=acc_name,
                    transactions=transactions,
                    parse_confidence=confidence
                ))

        return accounts

    def _detect_doc_type(self, text: str) -> str:
        text_lower = text.lower()
        if 'plaid' in text_lower or 'inflow' in text_lower or 'outflow' in text_lower:
            return 'plaid_report'
        if 'statement' in text_lower or 'previous balance' in text_lower:
            return 'bank_statement'
        return 'unknown'

    def _detect_year(self, text: str) -> int:
        years = re.findall(r'20[2-3]\d', text)
        if years:
            from collections import Counter
            return int(Counter(years).most_common(1)[0][0])
        return datetime.now().year

    def _find_accounts(self, text: str) -> List[Tuple[str, str, str]]:
        if self._doc_type == 'plaid_report':
            return self._find_plaid_accounts(text)
        return self._find_bank_accounts(text)

    def _find_plaid_accounts(self, text: str) -> List[Tuple[str, str, str]]:
        accounts = []
        summary_pattern = r'(?:Centra Credit Union|Bank)\s+(\w[\w\s]+?)\s+(\d{4,})\s+\$[\d,]+\.\d{2}'
        summary_matches = re.findall(summary_pattern, text, re.IGNORECASE)

        account_info = {name.strip(): mask for name, mask in summary_matches}

        sections = re.split(r'Account Overview', text, flags=re.IGNORECASE)

        for i, section in enumerate(sections[1:], 1):
            name_match = re.search(r'ACCOUNT NAME:\s*(.+?)(?:\s+ACCOUNT|\n)', section)
            if name_match:
                acc_name = name_match.group(1).strip()
                acc_id = account_info.get(acc_name, f"acc_{i}")

                history_match = re.search(r'Account History(.+?)(?=Account Overview|\Z)',
                                         section, re.DOTALL | re.IGNORECASE)
                if history_match:
                    accounts.append((acc_id, acc_name, history_match.group(1)))

        return accounts

    def _find_bank_accounts(self, text: str) -> List[Tuple[str, str, str]]:
        acc_match = re.search(r'(?:CHECKING|SAVINGS|ACCOUNT)[:\s]*(\d{6,})', text, re.IGNORECASE)
        if acc_match:
            acc_id = acc_match.group(1)
            name_match = re.search(r'(BUSINESS\s+\w+\s+\d*\s*(?:CHECKING|SAVINGS))', text, re.IGNORECASE)
            acc_name = name_match.group(1).title() if name_match else "Checking"
            return [(acc_id, acc_name, text)]
        return []

    def _parse_transactions(self, text: str) -> List[Transaction]:
        transactions = []
        current_section = None

        for line in text.split('\n'):
            line = line.strip()
            if not line or len(line) < 8:
                continue

            lower = line.lower()
            if any(kw in lower for kw in ['deposit', 'credit', 'inflow']):
                current_section = 'credit'
            elif any(kw in lower for kw in ['withdrawal', 'debit', 'outflow', 'check']):
                current_section = 'debit'

            tx = self._parse_line(line, current_section)
            if tx:
                transactions.append(tx)

        return self._deduplicate(transactions)

    def _parse_line(self, line: str, section: Optional[str]) -> Optional[Transaction]:
        skip_patterns = ['date', 'description', 'balance', 'page', 'total', 'summary']
        if any(p in line.lower() for p in skip_patterns):
            return None

        # Plaid format
        tx = self._parse_plaid_line(line)
        if tx:
            return tx

        # Bank format
        return self._parse_bank_line(line, section)

    def _parse_plaid_line(self, line: str) -> Optional[Transaction]:
        pattern = (
            r'^([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\s+'
            r'(.+?)\s+'
            r'(\$?[\d,]+\.\d{2}|---)\s+'
            r'(\$?[\d,]+\.\d{2}|---)'
        )
        match = re.match(pattern, line)
        if match:
            date_str, description, inflow, outflow = match.groups()
            tx_date = self._parse_date(date_str)
            if not tx_date:
                return None

            if inflow and inflow != '---':
                amount = float(inflow.replace('$', '').replace(',', ''))
                tx_type = TransactionType.CREDIT
            elif outflow and outflow != '---':
                amount = float(outflow.replace('$', '').replace(',', ''))
                tx_type = TransactionType.DEBIT
            else:
                return None

            if amount <= 0:
                return None

            return Transaction(
                date=tx_date,
                description=description.strip()[:200],
                amount=amount,
                type=tx_type
            )
        return None

    def _parse_bank_line(self, line: str, section: Optional[str]) -> Optional[Transaction]:
        pattern = r'^(\d{1,2}/\d{1,2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$'
        match = re.match(pattern, line)
        if match:
            date_str, description, amount_str = match.groups()
            tx_date = self._parse_date(date_str)
            if not tx_date:
                return None

            amount = float(amount_str.replace(',', ''))
            if amount <= 0:
                return None

            if section == 'credit':
                tx_type = TransactionType.CREDIT
            elif section == 'debit':
                tx_type = TransactionType.DEBIT
            else:
                tx_type = TransactionType.DEBIT

            return Transaction(
                date=tx_date,
                description=description.strip()[:200],
                amount=amount,
                type=tx_type
            )
        return None

    def _parse_date(self, date_str: str) -> Optional[date]:
        if not date_str:
            return None

        formats = [
            ("%B %d, %Y", False), ("%B %d %Y", False), ("%b %d, %Y", False),
            ("%m/%d/%Y", False), ("%m/%d/%y", False), ("%m/%d", True),
        ]
        for fmt, needs_year in formats:
            try:
                parsed = datetime.strptime(date_str.strip(), fmt)
                if needs_year:
                    parsed = parsed.replace(year=self._year)
                return parsed.date()
            except ValueError:
                continue
        return None

    def _calculate_confidence(self, acc_id: str, acc_name: str, transactions: List[Transaction]) -> float:
        score = 1.0
        if acc_id.startswith("acc_") or acc_id == "primary":
            score -= 0.2
        if acc_name in ("Primary", "Bank Account", "Checking", "Unknown"):
            score -= 0.15
        if len(transactions) < 5:
            score -= 0.2
        elif len(transactions) < 10:
            score -= 0.1
        if self._doc_type == "unknown":
            score -= 0.15
        return max(0.0, min(1.0, round(score, 2)))

    def _deduplicate(self, transactions: List[Transaction]) -> List[Transaction]:
        seen = set()
        unique = []
        for tx in transactions:
            key = (tx.date, tx.description[:30], tx.amount, tx.type)
            if key not in seen:
                seen.add(key)
                unique.append(tx)
        return sorted(unique, key=lambda x: x.date)


# ===============================
# Lender Matcher
# ===============================
class LenderMatcher:
    """Matches transactions to lenders using keywords."""

    def __init__(self, keywords_path: str = "keywords.json"):
        self.lender_keywords: Dict[str, List[str]] = {}
        self.transfer_keywords: List[str] = []
        self._all_keywords: List[str] = []
        self._keyword_to_lender: Dict[str, str] = {}
        self.match_threshold = 85

        self._load_keywords(keywords_path)

    def _load_keywords(self, path: str):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.lender_keywords = data.get("lenderKeywords", {})
            self.transfer_keywords = data.get("transferKeywords", [])

            for lender, keywords in self.lender_keywords.items():
                for kw in keywords:
                    upper_kw = kw.upper()
                    self._all_keywords.append(upper_kw)
                    self._keyword_to_lender[upper_kw] = lender
        except FileNotFoundError:
            print(f"Warning: Keywords file not found: {path}")

    def match_transactions(self, accounts: List[Account]) -> List[Account]:
        for account in accounts:
            for tx in account.transactions:
                self._categorize(tx)
        return accounts

    def _categorize(self, tx: Transaction):
        desc_upper = tx.description.upper()

        # Check for exact lender match first
        lender = self._match_lender(desc_upper)

        if lender:
            if tx.type == TransactionType.CREDIT:
                tx.category = TransactionCategory.LENDER_CREDIT
            else:
                tx.category = TransactionCategory.LENDER_PAYMENT
            tx.matched_lender = lender
            return

        # Check for transfer
        if self._is_transfer(tx.description):
            tx.category = TransactionCategory.TRANSFER_IN if tx.type == TransactionType.CREDIT else TransactionCategory.TRANSFER_OUT
            return

        # Default
        tx.category = TransactionCategory.DEPOSIT if tx.type == TransactionType.CREDIT else TransactionCategory.WITHDRAWAL

    def _match_lender(self, desc_upper: str) -> Optional[str]:
        # Exact keyword match
        for lender, keywords in self.lender_keywords.items():
            for kw in keywords:
                if kw.upper() in desc_upper:
                    return lender

        # Fuzzy match (if available and description long enough)
        if HAS_FUZZY and self._all_keywords and len(desc_upper) >= 12:
            eligible = [k for k in self._all_keywords if len(k) <= len(desc_upper)]
            if eligible:
                result = process.extractOne(desc_upper, eligible, scorer=fuzz.ratio, score_cutoff=self.match_threshold)
                if result:
                    return self._keyword_to_lender.get(result[0])

        return None

    def _is_transfer(self, desc: str) -> bool:
        lower = desc.lower()
        return any(kw.lower() in lower for kw in self.transfer_keywords)


# ===============================
# Financial Aggregator
# ===============================
class FinancialAggregator:
    """Aggregates financial metrics."""

    def __init__(self, reference_date: Optional[date] = None):
        self.reference_date = reference_date or date.today()

    def aggregate(self, accounts: List[Account]) -> List[AccountSummary]:
        return [self._aggregate_account(acc) for acc in accounts]

    def _aggregate_account(self, account: Account) -> AccountSummary:
        if account.transactions:
            ref_date = max(tx.date for tx in account.transactions)
        else:
            ref_date = self.reference_date

        m_start, m_end = self._get_recent_full_month(account.transactions, ref_date)
        d90_start = ref_date - timedelta(days=90)

        monthly_txns = [tx for tx in account.transactions if m_start <= tx.date <= m_end]
        d90_txns = [tx for tx in account.transactions if d90_start <= tx.date <= ref_date]

        monthly = self._calculate_metrics(monthly_txns)
        aggregate = self._calculate_metrics(d90_txns)

        return AccountSummary(
            account_id=account.account_id,
            account_name=account.account_name,
            monthly_transfers_in=monthly["transfers_in"],
            monthly_internal_transfers=monthly["internal_transfers"],
            monthly_lender_credits=monthly["lender_credits"],
            monthly_lender_payments=monthly["lender_payments"],
            monthly_deposits=monthly["total_deposits"],
            monthly_unique_lenders=monthly["unique_lenders"],
            monthly_lender_names=monthly["lender_names"],
            aggregate_deposits=aggregate["total_deposits"],
            aggregate_lender_payments=aggregate["lender_payments"],
            aggregate_lender_credits=aggregate["lender_credits"],
            debt_service_ratio=monthly["debt_service_ratio"],
            parse_confidence=account.parse_confidence,
        )

    def _get_recent_full_month(self, transactions: List[Transaction], ref_date: date) -> Tuple[date, date]:
        if not transactions:
            first_of_current = ref_date.replace(day=1)
            last_of_prev = first_of_current - timedelta(days=1)
            return last_of_prev.replace(day=1), last_of_prev

        max_date = max(tx.date for tx in transactions)
        current_month_txns = [tx for tx in transactions
                             if tx.date.year == max_date.year and tx.date.month == max_date.month]

        has_late_month = any(tx.date.day >= 25 for tx in current_month_txns)

        if not has_late_month:
            first_of_current = max_date.replace(day=1)
            last_of_prev = first_of_current - timedelta(days=1)
            _, last_day = monthrange(last_of_prev.year, last_of_prev.month)
            return last_of_prev.replace(day=1), last_of_prev.replace(day=last_day)

        _, last_day = monthrange(max_date.year, max_date.month)
        return max_date.replace(day=1), max_date.replace(day=last_day)

    def _calculate_metrics(self, transactions: List[Transaction]) -> Dict:
        metrics = {
            "transfers_in": 0.0, "internal_transfers": 0.0, "lender_credits": 0.0,
            "lender_payments": 0.0, "total_deposits": 0.0, "total_withdrawals": 0.0,
            "unique_lenders": 0, "lender_names": [], "debt_service_ratio": 0.0
        }

        lenders = set()

        for tx in transactions:
            if tx.category == TransactionCategory.TRANSFER_IN:
                metrics["transfers_in"] += tx.amount
                metrics["internal_transfers"] += tx.amount
                metrics["total_deposits"] += tx.amount
            elif tx.category == TransactionCategory.LENDER_CREDIT:
                metrics["lender_credits"] += tx.amount
                metrics["transfers_in"] += tx.amount
                metrics["total_deposits"] += tx.amount
                if tx.matched_lender:
                    lenders.add(tx.matched_lender)
            elif tx.category == TransactionCategory.LENDER_PAYMENT:
                metrics["lender_payments"] += tx.amount
                metrics["total_withdrawals"] += tx.amount
                if tx.matched_lender:
                    lenders.add(tx.matched_lender)
            elif tx.category == TransactionCategory.DEPOSIT:
                metrics["total_deposits"] += tx.amount
            elif tx.category == TransactionCategory.WITHDRAWAL:
                metrics["total_withdrawals"] += tx.amount

        metrics["unique_lenders"] = len(lenders)
        metrics["lender_names"] = sorted(lenders)

        if metrics["total_deposits"] > 0:
            metrics["debt_service_ratio"] = round(metrics["lender_payments"] / metrics["total_deposits"], 4)

        for key in ["transfers_in", "internal_transfers", "lender_credits", "lender_payments", "total_deposits"]:
            metrics[key] = round(metrics[key], 2)

        return metrics


# ===============================
# LLM Summary Generator
# ===============================
class SummaryGenerator:
    """Generates natural language summaries using LLM."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, summaries: List[AccountSummary]) -> List[AccountSummary]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")

            for s in summaries:
                prompt = f"""Summarize this bank account activity concisely (2-3 sentences):

Account: {s.account_name} (ending in {s.account_id})
Monthly inbound transfers: ${s.monthly_transfers_in:,.2f}
Monthly lender credits: ${s.monthly_lender_credits:,.2f}
Monthly deposits (total): ${s.monthly_deposits:,.2f}
Monthly payments to lenders: ${s.monthly_lender_payments:,.2f}
Unique lenders: {s.monthly_unique_lenders} ({', '.join(s.monthly_lender_names) or 'none'})
90-day total deposits: ${s.aggregate_deposits:,.2f}
90-day total lender payments: ${s.aggregate_lender_payments:,.2f}

Focus on key financial activity. Be factual and concise."""

                response = client.chat.completions.create(
                    model="grok-2-latest",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3
                )
                s.summary = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  LLM error: {e}")
            summaries = _basic_summaries(summaries)

        return summaries


# ===============================
# Main Pipeline
# ===============================
def _basic_summaries(summaries: List[AccountSummary]) -> List[AccountSummary]:
    for s in summaries:
        parts = []
        if s.monthly_deposits > 0:
            parts.append(f"Monthly deposits: ${s.monthly_deposits:,.2f}.")
        if s.monthly_lender_payments > 0:
            lenders = ", ".join(s.monthly_lender_names[:3]) or "lenders"
            parts.append(f"Lender payments to {lenders}: ${s.monthly_lender_payments:,.2f}.")
        if s.monthly_lender_credits > 0:
            parts.append(f"Lender credits: ${s.monthly_lender_credits:,.2f}.")
        if s.aggregate_deposits > 0:
            parts.append(f"90-day total deposits: ${s.aggregate_deposits:,.2f}.")
        s.summary = " ".join(parts) if parts else "No significant activity."
    return summaries


def extract(pdf_path: str, keywords_path: str = "keywords.json", use_llm_summary: bool = False) -> dict:
    """Fast extraction pipeline."""
    pdf_path = Path(pdf_path)
    print(f"Processing: {pdf_path.name}")

    # Stage 1: Parse
    print("  Parsing transactions...")
    parser = SmartParser()
    accounts = parser.parse_pdf(pdf_path)
    total_txns = sum(len(a.transactions) for a in accounts)
    print(f"    Found {total_txns} transactions in {len(accounts)} account(s)")

    # Stage 2: Match lenders
    print("  Matching lenders...")
    matcher = LenderMatcher(keywords_path=keywords_path)
    accounts = matcher.match_transactions(accounts)
    matched = sum(1 for a in accounts for t in a.transactions if t.matched_lender)
    print(f"    Matched {matched} lender transactions")

    # Stage 3: Aggregate
    print("  Aggregating metrics...")
    aggregator = FinancialAggregator()
    summaries = aggregator.aggregate(accounts)

    # Stage 4: Summaries
    if use_llm_summary:
        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROQ_API_KEY")
        if api_key:
            print("  Generating LLM summaries...")
            generator = SummaryGenerator(api_key=api_key)
            summaries = generator.generate(summaries)
        else:
            print("  No API key - using basic summaries")
            summaries = _basic_summaries(summaries)
    else:
        print("  Generating basic summaries...")
        summaries = _basic_summaries(summaries)

    # Build result
    result = {
        "extraction_date": date.today().isoformat(),
        "source_file": pdf_path.name,
        "accounts": []
    }

    for s in summaries:
        has_lender_activity = s.monthly_unique_lenders > 0 or s.monthly_lender_credits > 0
        transfers = s.monthly_lender_credits if has_lender_activity else s.monthly_internal_transfers

        result["accounts"].append({
            "accountId": s.account_id,
            "accountName": s.account_name,
            "transfers": round(transfers, 2),
            "internalTransfers": s.monthly_internal_transfers,
            "lenderCredits": s.monthly_lender_credits,
            "payments": s.monthly_lender_payments,
            "lenders": s.monthly_unique_lenders,
            "lenderNames": s.monthly_lender_names,
            "deposits": s.monthly_deposits,
            "summary": s.summary,
            "aggregateDeposits": s.aggregate_deposits,
            "aggregatePayments": s.aggregate_lender_payments,
            "debtServiceRatio": s.debt_service_ratio,
            "parseConfidence": s.parse_confidence,
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="Fast PDF transaction extractor")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--llm", action="store_true", help="Use LLM for summaries")
    parser.add_argument("-k", "--keywords", default="keywords.json", help="Keywords file")

    args = parser.parse_args()
    result = extract(args.pdf, args.keywords, args.llm)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    for acc in result["accounts"]:
        print(f"\n{acc['accountName']} ({acc['accountId']})")
        print(f"  Deposits:               ${acc['deposits']:,.2f}")
        print(f"  Internal Transfers:     ${acc['internalTransfers']:,.2f}")
        print(f"  Lender Credits:         ${acc['lenderCredits']:,.2f}")
        print(f"  Payments (to lenders):  ${acc['payments']:,.2f}")
        print(f"  Lenders:                {', '.join(acc['lenderNames']) or 'none'}")
        print(f"  Debt Service Ratio:     {acc['debtServiceRatio']:.2%}")
        print(f"  Parse Confidence:       {acc['parseConfidence']:.0%}")
        print(f"  --- 90-Day Aggregates ---")
        print(f"  90-Day Deposits:        ${acc['aggregateDeposits']:,.2f}")
        print(f"  90-Day Lender Payments: ${acc['aggregatePayments']:,.2f}")
        print(f"\n  Summary: {acc['summary']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {args.output}")

    return result


if __name__ == "__main__":
    main()
