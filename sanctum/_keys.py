"""
Ed25519 key management and KEYREGISTRY.toml interface.
"""

import base64
import hashlib
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
    load_pem_private_key,
    load_pem_public_key,
)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def _pub_raw(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def fingerprint(pub_bytes: bytes) -> str:
    """Return ``ed25519:<base64url-no-pad(sha256(pub_bytes)[:16])>`` fingerprint.

    The first 16 bytes of sha256(public_key_raw) give a compact 128-bit
    identifier.  URL-safe base64 without padding is used so the fingerprint
    is safe in filenames, TOML values, and command-line arguments.
    """
    digest = hashlib.sha256(pub_bytes).digest()[:16]
    return "ed25519:" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def generate_keypair(sk_path: Path, pk_path: Path) -> str:
    """Generate an Ed25519 keypair, write PEM files, and return fingerprint."""
    private_key = Ed25519PrivateKey.generate()
    sk_path.write_bytes(
        private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    )
    pk_path.write_bytes(
        private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    return fingerprint(_pub_raw(private_key))


def load_private_key(sk_path: Path) -> Ed25519PrivateKey:
    return load_pem_private_key(sk_path.read_bytes(), password=None)


def fingerprint_from_sk(sk_path: Path) -> str:
    return fingerprint(_pub_raw(load_private_key(sk_path)))


def fingerprint_from_pk(pk_path: Path) -> str:
    pub = load_pem_public_key(pk_path.read_bytes())
    return fingerprint(pub.public_bytes(Encoding.Raw, PublicFormat.Raw))


# ---------------------------------------------------------------------------
# KEYREGISTRY.toml helpers
# ---------------------------------------------------------------------------

def load_registry(registry_path: Path) -> dict:
    with open(registry_path, "rb") as fh:
        return tomllib.load(fh)


def _write_registry(registry_path: Path, reg: dict) -> None:
    """Serialise *reg* back to TOML (hand-rolled to avoid extra deps)."""
    lines = [
        "[meta]",
        f"version = {reg['meta']['version']}",
        f"frozen  = {str(reg['meta']['frozen']).lower()}",
        "",
    ]
    for key in reg.get("keys", []):
        lines += [
            "[[keys]]",
            f'fingerprint = "{key["fingerprint"]}"',
            f'status      = "{key["status"]}"',
            f'owner       = "{key["owner"]}"',
            f'created     = "{key["created"]}"',
            f'retired     = "{key.get("retired", "")}"',
            f'revoked     = "{key.get("revoked", "")}"',
            f'reason      = "{key.get("reason", "")}"',
            "",
        ]
    registry_path.write_text("\n".join(lines), encoding="utf-8")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def registry_add_key(registry_path: Path, fp: str, owner: str) -> None:
    if registry_path.exists():
        reg = load_registry(registry_path)
    else:
        reg = {"meta": {"version": 1, "frozen": False}, "keys": []}
    reg.setdefault("keys", [])
    reg["keys"].append({
        "fingerprint": fp,
        "status": "active",
        "owner": owner,
        "created": _now_utc(),
        "retired": "",
        "revoked": "",
        "reason": "",
    })
    _write_registry(registry_path, reg)


def registry_retire_key(registry_path: Path, fp: str) -> None:
    reg = load_registry(registry_path)
    now = _now_utc()
    for k in reg.get("keys", []):
        if k["fingerprint"] == fp and k["status"] == "active":
            k["status"] = "retired"
            k["retired"] = now
    _write_registry(registry_path, reg)


def registry_revoke_key(registry_path: Path, fp: str, reason: str) -> None:
    reg = load_registry(registry_path)
    now = _now_utc()
    for k in reg.get("keys", []):
        if k["fingerprint"] == fp and k["status"] in ("active", "retired"):
            k["status"] = "revoked"
            k["revoked"] = now
            k["reason"] = reason
    _write_registry(registry_path, reg)


def registry_find_key(reg: dict, fp: str) -> dict | None:
    for k in reg.get("keys", []):
        if k["fingerprint"] == fp:
            return k
    return None
