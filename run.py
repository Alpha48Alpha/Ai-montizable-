#!/usr/bin/env python3
"""
AI-Monetizable — Unified Runner
================================
Runs all content-generation modules in sequence:
  1. Movie Engine  — generates a 10-section animation production package
  2. Shoe Demo     — generates AI-powered shoe product packages

Run:
    python run.py
"""

import movie_engine
import shoe_demo


def main() -> None:
    print("=" * 70)
    print("  AI-MONETIZABLE — FULL CONTENT GENERATION SUITE")
    print("=" * 70)
    print()

    modules = [
        ("Movie Engine", movie_engine.main),
        ("Shoe Demo", shoe_demo.main),
    ]
    results = {}

    for i, (label, fn) in enumerate(modules, start=1):
        print(f"[ {i} / {len(modules)} ]  {label}")
        print("-" * 70)
        try:
            fn()
            results[label] = "OK"
        except Exception as exc:
            results[label] = f"FAILED — {exc}"
            print(f"ERROR in {label}: {exc}")
        print()

    print("=" * 70)
    all_ok = all(v == "OK" for v in results.values())
    for label, status in results.items():
        print(f"  {label}: {status}")
    if all_ok:
        print()
        print("  All modules completed successfully.")
    print("=" * 70)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
