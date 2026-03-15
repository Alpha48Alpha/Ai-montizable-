"""
Evidence export: pack the run bundle into an auditor evidence.zip.
"""

import zipfile
from pathlib import Path


def export_evidence(run_dir: Path, output_format: str = "evidence.zip") -> Path:
    """Pack provenance, signatures, and tarballs into an auditor bundle.

    The bundle is written to ``<run_dir>/<output_format>`` and also
    includes ``KEYREGISTRY.toml`` from the working directory (if present).
    """
    out_path = run_dir / output_format

    candidates = [
        run_dir / "run.provenance.json",
        Path(str(run_dir / "run.provenance.json") + ".sig"),
        run_dir / "run.traces.tar.gz",
        Path(str(run_dir / "run.traces.tar.gz") + ".sig"),
        run_dir / "run.artifacts.tar.gz",
        Path(str(run_dir / "run.artifacts.tar.gz") + ".sig"),
        Path("KEYREGISTRY.toml"),
    ]

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in candidates:
            if path.exists():
                zf.write(path, path.name)

    included = [p.name for p in candidates if p.exists()]
    print(f"✅  Evidence bundle written: {out_path}")
    print(f"   Included: {', '.join(included)}")
    return out_path
