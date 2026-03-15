"""
Ed25519 signing: sign provenance JSON and tarballs; update key fingerprint.
"""

from pathlib import Path

from ._keys import load_private_key, fingerprint_from_sk
from ._provenance import load_provenance, save_provenance


def sign_file(sk_path: Path, target: Path) -> Path:
    """Sign *target* with the private key at *sk_path*.

    The detached signature is written to ``<target>.sig`` and returned.
    """
    private_key = load_private_key(sk_path)
    sig = private_key.sign(target.read_bytes())
    sig_path = Path(str(target) + ".sig")
    sig_path.write_bytes(sig)
    return sig_path


def sign_run(run_dir: Path, sk_path: Path) -> None:
    """Sign provenance.json and both tarballs; record fingerprint in provenance."""
    fp = fingerprint_from_sk(sk_path)

    prov_path = run_dir / "run.provenance.json"
    prov = load_provenance(prov_path)
    prov["signing"]["key_fingerprint"] = fp
    save_provenance(prov, prov_path)

    sign_file(sk_path, prov_path)

    for name in ("run.traces.tar.gz", "run.artifacts.tar.gz"):
        tar_path = run_dir / name
        if tar_path.exists():
            sign_file(sk_path, tar_path)
