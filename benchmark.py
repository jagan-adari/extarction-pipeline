#!/usr/bin/env python3
"""
Benchmark: Compare Agentic vs Vision extractors
Runs each extractor multiple times to measure consistency and reliability.
"""

import time
import json
from pathlib import Path

from agentic_extractor import agentic_extract
from vision_extractor import vision_extract


def benchmark(pdf_path: str, runs: int = 3):
    """Run both extractors multiple times and compare results."""
    pdf_path = Path(pdf_path)
    print(f"=" * 60)
    print(f"BENCHMARK: {pdf_path.name}")
    print(f"Runs per method: {runs}")
    print(f"=" * 60)

    agentic_results = []
    vision_results = []

    # Run agentic extractor
    print(f"\n--- AGENTIC EXTRACTOR ---")
    for i in range(runs):
        print(f"\nRun {i+1}/{runs}...")
        start = time.time()
        txns, error = agentic_extract(str(pdf_path))
        elapsed = round(time.time() - start, 2)
        agentic_results.append({
            "run": i + 1,
            "count": len(txns),
            "time": elapsed,
            "error": error,
            "transactions": txns
        })
        status = f"{len(txns)} txns" if not error else f"ERROR: {error[:50]}"
        print(f"  -> {status} in {elapsed}s")

    # Run vision extractor
    print(f"\n--- VISION EXTRACTOR ---")
    for i in range(runs):
        print(f"\nRun {i+1}/{runs}...")
        start = time.time()
        txns, error = vision_extract(str(pdf_path))
        elapsed = round(time.time() - start, 2)
        vision_results.append({
            "run": i + 1,
            "count": len(txns),
            "time": elapsed,
            "error": error,
            "transactions": txns
        })
        status = f"{len(txns)} txns" if not error else f"ERROR: {error[:50]}"
        print(f"  -> {status} in {elapsed}s")

    # Analysis
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")

    def analyze(results, name):
        counts = [r["count"] for r in results]
        times = [r["time"] for r in results]
        errors = [r["error"] for r in results if r["error"]]

        success_rate = ((len(results) - len(errors)) / len(results)) * 100
        avg_count = sum(counts) / len(counts) if counts else 0
        avg_time = sum(times) / len(times) if times else 0
        count_variance = max(counts) - min(counts) if counts else 0

        print(f"\n{name}:")
        print(f"  Success rate: {success_rate:.0f}%")
        print(f"  Avg transactions: {avg_count:.1f}")
        print(f"  Consistency: {count_variance} variance (max-min)")
        print(f"  Avg time: {avg_time:.1f}s")
        print(f"  Individual runs: {[r['count'] for r in results]}")

        return {
            "success_rate": success_rate,
            "avg_count": avg_count,
            "count_variance": count_variance,
            "avg_time": avg_time,
            "runs": counts
        }

    agentic_stats = analyze(agentic_results, "AGENTIC")
    vision_stats = analyze(vision_results, "VISION")

    # Recommendation
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATION")
    print(f"{'=' * 60}")

    if agentic_stats["success_rate"] == 0:
        print(f"-> Use VISION (Agentic failed completely)")
    elif agentic_stats["success_rate"] < 100 and vision_stats["success_rate"] == 100:
        print(f"-> Use VISION (more reliable)")
    elif agentic_stats["count_variance"] > 5 and vision_stats["count_variance"] <= 5:
        print(f"-> Use VISION (more consistent)")
    elif agentic_stats["avg_count"] > 0 and vision_stats["avg_count"] > 0:
        diff = abs(agentic_stats["avg_count"] - vision_stats["avg_count"])
        if diff > 5:
            winner = "VISION" if vision_stats["avg_count"] > agentic_stats["avg_count"] else "AGENTIC"
            print(f"-> Use {winner} (extracts more transactions)")
        elif agentic_stats["avg_time"] < vision_stats["avg_time"] * 0.5:
            print(f"-> Use AGENTIC (2x+ faster, similar accuracy)")
        else:
            print(f"-> Both viable - AGENTIC for speed, VISION for reliability")
    else:
        print(f"-> Results inconclusive")

    return {
        "pdf": str(pdf_path),
        "agentic": agentic_stats,
        "vision": vision_stats
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF to benchmark")
    parser.add_argument("-n", "--runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()

    benchmark(args.pdf, args.runs)
