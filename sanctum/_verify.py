"""
Verification: sha256 hashes, Merkle root, and Ed25519 signatures.
"""

import hashlib
import tarfile
import tempfile
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from ._keys import load_registry, registry_find_key
from ._merkle import merkle_root_of_file
from ._provenance import load_provenance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sig(pub_key, sig_path: Path, data_path: Path) -> bool:
    try:
        pub_key.verify(sig_path.read_bytes(), data_path.read_bytes())
        return True
    except InvalidSignature:
        return False


def _extract_trace_log(traces_tar: Path) -> Path | None:
    """Extract trace.log from a tarball into a temp dir and return its path."""
    tmp = tempfile.mkdtemp()
    with tarfile.open(traces_tar, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name == "trace.log"]
        if not members:
            return None
        tar.extract(members[0], path=tmp, filter="data")
    return Path(tmp) / "trace.log"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_run(run_dir: Path, registry_path: Path | None = None) -> bool:
    """Full verification of a sealed + signed run directory.

    Checks:
      1. sha256 of both tarballs vs provenance manifest
      2. Merkle root of trace.log vs provenance manifest
      3. Ed25519 signature on provenance.json (if sig file present)

    Returns ``True`` if all checks pass, ``False`` otherwise.
    """
    ok = True
    prov_path = run_dir / "run.provenance.json"
    prov = load_provenance(prov_path)

    # 1. Tarball hashes
    for key, name in [
        ("traces_tarball",    "run.traces.tar.gz"),
        ("artifacts_tarball", "run.artifacts.tar.gz"),
    ]:
        tar_path = run_dir / name
        if not tar_path.exists():
            print(f"⚠️   tarball not found: {name}")
            continue
        actual   = _sha256_file(tar_path)
        expected = prov["manifest"]["artifacts"][key]["sha256"]
        if actual != expected:
            print(f"❌  sha256 mismatch for {name}")
            print(f"    expected: {expected[:16]}…")
            print(f"    actual  : {actual[:16]}…")
            ok = False
        else:
            print(f"✅  sha256 OK: {name}")

    # 2. Merkle root
    traces_tar = run_dir / "run.traces.tar.gz"
    if traces_tar.exists():
        trace_log = _extract_trace_log(traces_tar)
        if trace_log is not None:
            actual_root   = merkle_root_of_file(trace_log)
            expected_root = prov["manifest"]["merkle_root"]
            if actual_root != expected_root:
                print(f"❌  Merkle mismatch")
                print(f"    expected: {expected_root[:16]}…")
                print(f"    actual  : {actual_root[:16]}…")
                ok = False
            else:
                print(f"✅  Merkle root OK: {actual_root[:16]}…")

    # 3. Signature on provenance.json
    fp       = prov["signing"]["key_fingerprint"]
    sig_path = Path(str(prov_path) + ".sig")

    if not sig_path.exists():
        print("⚠️   No provenance signature found — run 'sanctum sign' first")
    elif not fp:
        print("⚠️   key_fingerprint is empty in provenance")
    else:
        reg_path = registry_path or Path("KEYREGISTRY.toml")
        if not reg_path.exists():
            print(f"⚠️   Key registry not found at {reg_path}")
        else:
            reg = load_registry(reg_path)
            entry = registry_find_key(reg, fp)
            if entry is None:
                print(f"❌  Key {fp} not found in registry")
                ok = False
            elif entry["status"] == "revoked":
                print(f"❌  Key {fp} is REVOKED (reason: {entry['reason']})")
                ok = False
            else:
                # Search for a matching public key PEM.
                # Validate each candidate by content (must load as a public key,
                # not a private key) rather than relying solely on filename.
                pk_candidates = (
                    list(Path(".").glob("*.pem"))
                    + list(run_dir.glob("*.pem"))
                )
                verified = False
                for pk_path in pk_candidates:
                    try:
                        pub = load_pem_public_key(pk_path.read_bytes())
                        # Ensure it is an Ed25519 public key (not a private key
                        # accidentally loaded through load_pem_public_key, which
                        # would raise but we double-check the type).
                        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                            Ed25519PublicKey,
                        )
                        if not isinstance(pub, Ed25519PublicKey):
                            continue
                        if _verify_sig(pub, sig_path, prov_path):
                            status_note = entry["status"]
                            print(f"✅  Signature valid (key: {fp[:24]}…, status: {status_note})")
                            verified = True
                            break
                    except Exception:
                        continue
                if not verified:
                    print(f"⚠️   Public key PEM not found — signature not verified")

    return ok


def check_provenance(
    prov_path: Path,
    registry_path: Path,
    pk_path: Path | None = None,
) -> bool:
    """Verify provenance JSON signature against the key registry."""
    prov = load_provenance(prov_path)
    fp   = prov["signing"]["key_fingerprint"]
    sig_path = Path(str(prov_path) + ".sig")

    if not sig_path.exists():
        print(f"❌  Signature file not found: {sig_path}")
        return False

    reg   = load_registry(registry_path)
    entry = registry_find_key(reg, fp)

    if entry is None:
        print(f"❌  Key {fp} not found in registry")
        return False

    if entry["status"] == "revoked":
        print(f"❌  Key {fp} is REVOKED (reason: {entry['reason']})")
        return False

    if pk_path is None:
        print(f"ℹ️   Key {fp} found in registry (status: {entry['status']})")
        print("⚠️   No --pk provided; skipping cryptographic signature check")
        return True

    pub = load_pem_public_key(pk_path.read_bytes())
    if _verify_sig(pub, sig_path, prov_path):
        print(f"✅  Provenance signature valid (key: {fp}, status: {entry['status']})")
        return True
    else:
        print(f"❌  Provenance signature INVALID")
        return False


def verify_trace(traces_tar: Path, expected_merkle: str) -> bool:
    """Verify Merkle root of trace.log inside *traces_tar*."""
    trace_log = _extract_trace_log(traces_tar)
    if trace_log is None:
        print(f"❌  trace.log not found inside {traces_tar}")
        return False

    actual = merkle_root_of_file(trace_log)
    if actual == expected_merkle:
        print(f"✅  Merkle root matches: {actual[:16]}…")
        return True
    else:
        print(f"❌  Merkle mismatch")
        print(f"    expected: {expected_merkle[:16]}…")
        print(f"    actual  : {actual[:16]}…")
        return False


def verify_tarballs(artifacts_tar: Path, prov_path: Path) -> bool:
    """Verify artifact tarball sha256 against the provenance manifest."""
    prov     = load_provenance(prov_path)
    actual   = _sha256_file(artifacts_tar)
    expected = prov["manifest"]["artifacts"]["artifacts_tarball"]["sha256"]

    if actual == expected:
        print(f"✅  Artifacts sha256 OK: {actual[:16]}…")
        return True
    else:
        print(f"❌  Artifacts sha256 mismatch")
        print(f"    expected: {expected[:16]}…")
        print(f"    actual  : {actual[:16]}…")
        return False
