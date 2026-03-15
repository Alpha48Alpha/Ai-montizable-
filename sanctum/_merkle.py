"""
Deterministic Merkle tree over trace.log lines.

Each leaf  = sha256(line_bytes).
Folding    : pairwise sha256(left || right); duplicate last leaf if count is odd.
"""

import hashlib
from pathlib import Path


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def compute_merkle_root(lines: list) -> str:
    """Return hex Merkle root for a list of raw line byte-strings.

    Each element should include its trailing newline exactly as it appears
    in the file.  Returns ``""`` for an empty list.
    """
    if not lines:
        return ""

    nodes = [_sha256(line) for line in lines]

    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            # Duplicate the last leaf so every level has an even count.
            # This matches the Bitcoin/standard Merkle construction and
            # ensures determinism without introducing a sentinel zero-node.
            nodes.append(nodes[-1])
        nodes = [_sha256(nodes[i] + nodes[i + 1]) for i in range(0, len(nodes), 2)]

    return nodes[0].hex()


def merkle_root_of_file(path: Path) -> str:
    """Compute Merkle root from a trace-log file on disk."""
    raw = path.read_bytes()
    if not raw:
        return ""
    # Normalise: every line ends with exactly one newline
    lines = [line + b"\n" for line in raw.splitlines()]
    return compute_merkle_root(lines)
