"""
Sealing: create tarballs, compute sha256 hashes and Merkle root,
then update the provenance manifest in-place.
"""

import hashlib
import tarfile
from pathlib import Path

from ._merkle import merkle_root_of_file
from ._provenance import load_provenance, save_provenance


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_tarball(dest: Path, source_dir: Path) -> None:
    with tarfile.open(dest, "w:gz") as tar:
        for child in sorted(source_dir.iterdir()):
            tar.add(child, arcname=child.name)


def seal_run(run_dir: Path) -> dict:
    """Create tarballs, compute hashes + Merkle, and update provenance.

    Returns the updated provenance dict.
    """
    prov_path = run_dir / "run.provenance.json"
    prov = load_provenance(prov_path)

    traces_dir   = run_dir / "traces"
    artifacts_dir = run_dir / "artifacts"
    traces_tar   = run_dir / "run.traces.tar.gz"
    artifacts_tar = run_dir / "run.artifacts.tar.gz"

    # Ensure directories exist
    traces_dir.mkdir(parents=True, exist_ok=True)
    if not (traces_dir / "trace.log").exists():
        (traces_dir / "trace.log").touch()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _make_tarball(traces_tar, traces_dir)
    _make_tarball(artifacts_tar, artifacts_dir)

    # Merkle root over trace.log
    trace_log = traces_dir / "trace.log"
    merkle_root = merkle_root_of_file(trace_log)

    # sha256 of tarballs
    traces_sha    = _sha256_file(traces_tar)
    artifacts_sha = _sha256_file(artifacts_tar)

    # Persist merkle_root.txt alongside trace.log
    (traces_dir / "merkle_root.txt").write_text(merkle_root + "\n", encoding="utf-8")

    # Update provenance
    prov["manifest"]["merkle_root"] = merkle_root
    prov["manifest"]["artifacts"]["traces_tarball"]["sha256"]    = traces_sha
    prov["manifest"]["artifacts"]["artifacts_tarball"]["sha256"] = artifacts_sha

    save_provenance(prov, prov_path)
    return prov
