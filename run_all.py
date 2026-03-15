#!/usr/bin/env python3
"""
run_all.py — Run all AI-Monetizable scripts
============================================
Executes every production-package generator in the repository in sequence:
  1. movie_engine.py  — Full animation movie production package
  2. shoe_demo.py     — AI-powered shoe product package

Run:
    python run_all.py
"""

import movie_engine
import shoe_demo


def main() -> None:
    print("=" * 70)
    print("  RUNNING ALL AI-MONETIZABLE SCRIPTS")
    print("=" * 70)
    print()

    scripts = [
        ("[1/2] Movie Engine", movie_engine.main),
        ("[2/2] Shoe Demo", shoe_demo.main),
    ]

    failed = []
    for label, fn in scripts:
        print(f"▶  {label}")
        print("-" * 70)
        try:
            fn()
        except Exception as exc:
            print(f"\n❌  {label} FAILED: {exc}\n")
            failed.append(label)
        print()

    print("=" * 70)
    if failed:
        print(f"  COMPLETED WITH ERRORS — {len(failed)} script(s) failed:")
        for name in failed:
            print(f"    • {name}")
        raise SystemExit(1)
    else:
        print("  ALL SCRIPTS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
