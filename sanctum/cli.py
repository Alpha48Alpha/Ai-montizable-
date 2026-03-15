"""
Sanctum CLI — AI run provenance, signing, and verification.

Usage
-----
  sanctum keys init   --owner ORG:TEAM --algo ed25519
  sanctum keys list
  sanctum keys rotate --owner ORG:TEAM
  sanctum keys revoke --fingerprint ed25519:<...> --reason "compromise"

  sanctum run     --dataset-hash sha256:... --model-hash sha256:... \\
                  --git-commit $(git rev-parse --short HEAD) \\
                  --sbom sbom.spdx.json --out ./runs/20250905-1432

  sanctum seal    --dir ./runs/20250905-1432
  sanctum sign    --dir ./runs/20250905-1432 --key demo_sk.pem

  sanctum verify_run       --dir ./runs/20250905-1432
  sanctum check_provenance --provenance ./runs/.../run.provenance.json \\
                            --key-registry KEYREGISTRY.toml
  sanctum verify_trace     --traces ./runs/.../run.traces.tar.gz \\
                            --expected-merkle <hex>
  sanctum verify_tarballs  --artifacts ./runs/.../run.artifacts.tar.gz \\
                            --provenance ./runs/.../run.provenance.json

  sanctum export  --dir ./runs/20250905-1432 --format evidence.zip
"""

import sys
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Sanctum: cryptographic provenance and integrity for AI runs."""


# ---------------------------------------------------------------------------
# keys sub-group
# ---------------------------------------------------------------------------

@cli.group()
def keys():
    """Key management: init, list, rotate, revoke."""


@keys.command("init")
@click.option("--owner",    required=True, help="Key owner label (e.g. ORG:TEAM)")
@click.option("--algo",     default="ed25519", show_default=True,
              help="Signing algorithm (currently only ed25519)")
@click.option("--registry", default="KEYREGISTRY.toml", show_default=True)
@click.option("--sk-out",   default="demo_sk.pem",       show_default=True)
@click.option("--pk-out",   default="demo_pk.pem",        show_default=True)
def keys_init(owner, algo, registry, sk_out, pk_out):
    """Generate a new Ed25519 keypair and register it."""
    from ._keys import generate_keypair, registry_add_key

    fp = generate_keypair(Path(sk_out), Path(pk_out))
    registry_add_key(Path(registry), fp, owner)

    click.echo(f"✅  Keypair generated: {sk_out} / {pk_out}")
    click.echo(f"   Fingerprint : {fp}")
    click.echo(f"   Registered  : {registry}")


@keys.command("list")
@click.option("--registry", default="KEYREGISTRY.toml", show_default=True)
def keys_list(registry):
    """List all keys in the registry."""
    from ._keys import load_registry

    reg  = load_registry(Path(registry))
    rows = reg.get("keys", [])
    if not rows:
        click.echo("No keys registered.")
        return
    header = f"{'Fingerprint':<44} {'Status':<10} {'Owner':<22} Created"
    click.echo(header)
    click.echo("-" * len(header))
    for k in rows:
        click.echo(
            f"{k['fingerprint']:<44} {k['status']:<10} {k['owner']:<22} {k['created']}"
        )


@keys.command("rotate")
@click.option("--owner",    required=True)
@click.option("--registry", default="KEYREGISTRY.toml", show_default=True)
@click.option("--sk-out",   default=None)
@click.option("--pk-out",   default=None)
def keys_rotate(owner, registry, sk_out, pk_out):
    """Retire the active key for *owner* and generate a new one."""
    import re
    from ._keys import (
        load_registry, registry_retire_key,
        generate_keypair, registry_add_key,
    )

    reg_path = Path(registry)
    reg = load_registry(reg_path)
    for k in reg.get("keys", []):
        if k["owner"] == owner and k["status"] == "active":
            registry_retire_key(reg_path, k["fingerprint"])
            click.echo(f"   Retired: {k['fingerprint']}")

    slug = re.sub(r"[^a-z0-9]", "_", owner.lower())
    sk = Path(sk_out or f"{slug}_sk.pem")
    pk = Path(pk_out or f"{slug}_pk.pem")
    fp = generate_keypair(sk, pk)
    registry_add_key(reg_path, fp, owner)
    click.echo(f"✅  New key: {fp}")
    click.echo(f"   Files  : {sk} / {pk}")


@keys.command("revoke")
@click.option("--fingerprint", "fp", required=True, help="Key fingerprint to revoke")
@click.option("--reason",      default="", show_default=True)
@click.option("--registry",    default="KEYREGISTRY.toml", show_default=True)
def keys_revoke(fp, reason, registry):
    """Revoke a key permanently."""
    from ._keys import registry_revoke_key

    registry_revoke_key(Path(registry), fp, reason)
    click.echo(f"✅  Revoked: {fp}")
    if reason:
        click.echo(f"   Reason : {reason}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option("--dataset-hash",   default="", help="sha256:... of training dataset tarball")
@click.option("--dataset-name",   default="")
@click.option("--model-hash",     default="", help="sha256:... of model checkpoint")
@click.option("--model-name",     default="")
@click.option("--model-commit",   default="")
@click.option("--git-commit",     default="",
              help="Short git commit hash (auto-detected if omitted)")
@click.option("--sbom",           default=None, help="Path to SBOM file (spdx)")
@click.option("--out",            required=True, help="Output run directory")
def run_cmd(dataset_hash, dataset_name, model_hash, model_name,
            model_commit, git_commit, sbom, out):
    """Initialise a run directory with a provenance record."""
    import hashlib
    import subprocess
    from datetime import datetime, timezone

    from ._provenance import create_provenance, save_provenance

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces").mkdir(exist_ok=True)
    (out_dir / "traces" / "trace.log").touch()
    (out_dir / "artifacts").mkdir(exist_ok=True)

    # Auto-detect git commit
    if not git_commit:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            git_commit = "unknown"

    # Hash the SBOM file if provided
    sbom_hash   = ""
    sbom_format = "spdx"
    if sbom:
        sbom_path = Path(sbom)
        if sbom_path.exists():
            sbom_hash = "sha256:" + hashlib.sha256(sbom_path.read_bytes()).hexdigest()

    prov = create_provenance(
        git_commit=git_commit,
        dataset_name=dataset_name,
        dataset_hash=dataset_hash,
        model_name=model_name,
        model_commit=model_commit,
        model_hash=model_hash,
        sbom_format=sbom_format,
        sbom_hash=sbom_hash,
        now=datetime.now(timezone.utc),
    )

    prov_path = out_dir / "run.provenance.json"
    save_provenance(prov, prov_path)

    click.echo(f"✅  Run initialised: {out_dir}")
    click.echo(f"   run_id    : {prov['run_id']}")
    click.echo(f"   timestamp : {prov['timestamp']}")
    click.echo(f"   provenance: {prov_path}")


# ---------------------------------------------------------------------------
# seal
# ---------------------------------------------------------------------------

@cli.command("seal")
@click.option("--dir", "run_dir", required=True, help="Run directory to seal")
def seal_cmd(run_dir):
    """Create tarballs, compute sha256 + Merkle root, update provenance."""
    from ._seal import seal_run

    prov = seal_run(Path(run_dir))
    root = prov["manifest"]["merkle_root"]
    click.echo(f"✅  Sealed: {run_dir}")
    click.echo(f"   Merkle root : {root[:16]}{'…' if len(root) > 16 else ''}")


# ---------------------------------------------------------------------------
# sign
# ---------------------------------------------------------------------------

@cli.command("sign")
@click.option("--dir", "run_dir", required=True)
@click.option("--key",           required=True, help="Path to Ed25519 private key PEM")
def sign_cmd(run_dir, key):
    """Sign provenance.json and tarballs with an Ed25519 private key."""
    from ._sign import sign_run

    sign_run(Path(run_dir), Path(key))
    click.echo(f"✅  Signed: {run_dir}")


# ---------------------------------------------------------------------------
# verify_run
# ---------------------------------------------------------------------------

@cli.command("verify_run")
@click.option("--dir",      "run_dir",  required=True)
@click.option("--registry", default="KEYREGISTRY.toml", show_default=True)
def verify_run_cmd(run_dir, registry):
    """Full verification: sha256, Merkle root, and signature."""
    from ._verify import verify_run

    reg_path = Path(registry)
    ok = verify_run(Path(run_dir), reg_path if reg_path.exists() else None)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# check_provenance
# ---------------------------------------------------------------------------

@cli.command("check_provenance")
@click.option("--provenance",   required=True, help="Path to run.provenance.json")
@click.option("--key-registry", required=True, help="Path to KEYREGISTRY.toml")
@click.option("--pk",           default=None,  help="Public key PEM for sig verification")
def check_provenance_cmd(provenance, key_registry, pk):
    """Verify provenance JSON signature against the key registry."""
    from ._verify import check_provenance

    ok = check_provenance(
        Path(provenance),
        Path(key_registry),
        Path(pk) if pk else None,
    )
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# verify_trace
# ---------------------------------------------------------------------------

@cli.command("verify_trace")
@click.option("--traces",           required=True, help="Path to run.traces.tar.gz")
@click.option("--expected-merkle",  required=True, help="Expected Merkle root (hex)")
def verify_trace_cmd(traces, expected_merkle):
    """Verify the Merkle root of trace.log inside the traces tarball."""
    from ._verify import verify_trace

    ok = verify_trace(Path(traces), expected_merkle)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# verify_tarballs
# ---------------------------------------------------------------------------

@cli.command("verify_tarballs")
@click.option("--artifacts",  required=True, help="Path to run.artifacts.tar.gz")
@click.option("--provenance", required=True, help="Path to run.provenance.json")
def verify_tarballs_cmd(artifacts, provenance):
    """Verify the artifact tarball sha256 against the provenance manifest."""
    from ._verify import verify_tarballs

    ok = verify_tarballs(Path(artifacts), Path(provenance))
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command("export")
@click.option("--dir",    "run_dir", required=True)
@click.option("--format", "fmt",     default="evidence.zip", show_default=True,
              help="Output filename for the evidence bundle")
def export_cmd(run_dir, fmt):
    """Export an auditor evidence bundle (evidence.zip)."""
    from ._export import export_evidence

    export_evidence(Path(run_dir), fmt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
