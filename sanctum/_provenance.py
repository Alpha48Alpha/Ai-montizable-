"""
Provenance JSON record — creation, loading, and saving.

Schema
------
{
  "run_id":    "<YYYYMMDD>-<HHMM>-<git_commit_short>",
  "timestamp": "<ISO-8601 UTC>",
  "provenance": {
    "git_commit": "<short hash>",
    "dataset":    {"name": "", "hash": "sha256:<...>"},
    "model":      {"name": "", "commit": "", "hash": "sha256:<...>"},
    "sbom":       {"format": "spdx", "hash": "sha256:<...>"}
  },
  "manifest": {
    "merkle_root": "",
    "artifacts": {
      "traces_tarball":    {"path": "run.traces.tar.gz",    "sha256": ""},
      "artifacts_tarball": {"path": "run.artifacts.tar.gz", "sha256": ""}
    }
  },
  "quality_gate": {
    "status":  "pending",          # pending | passed | failed
    "metrics": {"accuracy": 0.0, "loss": 0.0, "drift": "none"}
  },
  "signing": {"key_fingerprint": ""}
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def _run_id(now: datetime, git_commit: str) -> str:
    """Format: YYYYMMDD-HHMM-<7-char git commit>.

    *now* must be a UTC-aware datetime; a naive datetime will raise ValueError.
    """
    if now.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M")
    slug = (git_commit or "unknown")[:7]
    return f"{date_part}-{time_part}-{slug}"


def create_provenance(
    *,
    git_commit: str = "",
    dataset_name: str = "",
    dataset_hash: str = "",
    model_name: str = "",
    model_commit: str = "",
    model_hash: str = "",
    sbom_format: str = "spdx",
    sbom_hash: str = "",
    now: datetime | None = None,
) -> dict:
    """Build and return a fresh provenance record (not yet written to disk)."""
    if now is None:
        now = datetime.now(timezone.utc)

    return {
        "run_id": _run_id(now, git_commit),
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "git_commit": git_commit,
            "dataset": {"name": dataset_name, "hash": dataset_hash},
            "model": {"name": model_name, "commit": model_commit, "hash": model_hash},
            "sbom": {"format": sbom_format, "hash": sbom_hash},
        },
        "manifest": {
            "merkle_root": "",
            "artifacts": {
                "traces_tarball":    {"path": "run.traces.tar.gz",    "sha256": ""},
                "artifacts_tarball": {"path": "run.artifacts.tar.gz", "sha256": ""},
            },
        },
        "quality_gate": {
            "status": "pending",
            "metrics": {"accuracy": 0.0, "loss": 0.0, "drift": "none"},
        },
        "signing": {"key_fingerprint": ""},
    }


def load_provenance(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_provenance(prov: dict, path: Path) -> None:
    path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
