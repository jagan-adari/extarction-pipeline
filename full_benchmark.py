#!/usr/bin/env python3
"""
Full Benchmark: Run both extractors on all PDFs in data folder.
"""

import time
import json
from pathlib import Path

from agentic_extractor import agentic_extract
from vision_extractor import vision_extract


def run_full_benchmark():
    data_dir = Path("data")
    pdfs = sorted(data_dir.glob("*.pdf"))

    print(f"=" * 80)
    print(f"FULL BENCHMARK - {len(pdfs)} PDFs")
    print(f"=" * 80)

    results = []

    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        print("-" * 60)

        # Agentic
        print("  Agentic: ", end="", flush=True)
        start = time.time()
        agentic_txns, agentic_err = agentic_extract(str(pdf))
        agentic_time = round(time.time() - start, 1)
        print(f"{len(agentic_txns)} txns in {agentic_time}s" + (f" (ERR: {agentic_err[:30]})" if agentic_err else ""))

        # Vision
        print("  Vision:  ", end="", flush=True)
        start = time.time()
        vision_txns, vision_err = vision_extract(str(pdf))
        vision_time = round(time.time() - start, 1)
        print(f"{len(vision_txns)} txns in {vision_time}s" + (f" (ERR: {vision_err[:30]})" if vision_err else ""))

        # Compare
        diff = len(vision_txns) - len(agentic_txns)
        if len(agentic_txns) == 0 and len(vision_txns) > 0:
            winner = "VISION (agentic failed)"
        elif abs(diff) <= 2:
            winner = "TIE" if agentic_time < vision_time else "TIE (vision faster)"
        elif diff > 0:
            winner = f"VISION (+{diff} txns)"
        else:
            winner = f"AGENTIC (+{-diff} txns)"

        print(f"  Winner:  {winner}")

        results.append({
            "pdf": pdf.name,
            "agentic_count": len(agentic_txns),
            "agentic_time": agentic_time,
            "agentic_error": agentic_err,
            "vision_count": len(vision_txns),
            "vision_time": vision_time,
            "vision_error": vision_err,
            "winner": winner
        })

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n{'PDF':<50} {'Agentic':>10} {'Vision':>10} {'Winner':<25}")
    print("-" * 95)

    agentic_wins = 0
    vision_wins = 0
    ties = 0
    agentic_failures = 0

    for r in results:
        print(f"{r['pdf'][:48]:<50} {r['agentic_count']:>10} {r['vision_count']:>10} {r['winner']:<25}")
        if "AGENTIC" in r['winner']:
            agentic_wins += 1
        elif "VISION" in r['winner']:
            vision_wins += 1
        else:
            ties += 1
        if r['agentic_count'] == 0 and r['vision_count'] > 0:
            agentic_failures += 1

    print("-" * 95)
    print(f"\nScoreboard:")
    print(f"  Agentic wins: {agentic_wins}")
    print(f"  Vision wins:  {vision_wins}")
    print(f"  Ties:         {ties}")
    print(f"  Agentic failures (0 txns when vision found some): {agentic_failures}")

    total_agentic_time = sum(r['agentic_time'] for r in results)
    total_vision_time = sum(r['vision_time'] for r in results)
    print(f"\nTotal time:")
    print(f"  Agentic: {total_agentic_time:.0f}s ({total_agentic_time/60:.1f} min)")
    print(f"  Vision:  {total_vision_time:.0f}s ({total_vision_time/60:.1f} min)")

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: benchmark_results.json")


if __name__ == "__main__":
    run_full_benchmark()
