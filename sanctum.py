#!/usr/bin/env python3
"""Sanctum entry-point script.

Run as:
    python sanctum.py <command> [options]

Or install with pip (setup.cfg / pyproject.toml) and use the ``sanctum``
console-script entry-point directly.
"""

from sanctum.cli import cli

if __name__ == "__main__":
    cli()
