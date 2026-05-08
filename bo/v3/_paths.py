"""Path-immutability guard for the v3 BO driver.

Centralises the list of read-only data prefixes from `.cursorrules` and
provides a single helper, :func:`assert_writable`, that the driver and its
sandboxed subprocess wrappers call before resolving any output path. The
driver fails fast (raises ``ValueError``) if a resolved path falls under
any protected prefix; this prevents a misconfigured ``--base-dir`` flag
from ever overwriting ablation, BO, baseline, preprocessing-QA, represents,
or per-ring corpus snapshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


def _norm(rel: str) -> Path:
    return (REPO_ROOT / rel).resolve()


PROTECTED_PREFIXES: tuple[Path, ...] = (
    _norm("data/ablation"),
    _norm("data/bo"),
    _norm("data/baseline"),
    _norm("data/preprocessing_qa"),
    _norm("data/represents"),
    _norm("logs/context_preprocessing_v1"),
    _norm("r4tun/data"),
    _norm("r4tun/references"),
    _norm("methods/plans/output"),
    _norm("stages"),
)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
        return True
    except ValueError:
        return False


_TUNNEL_ID_RE = __import__("re").compile(r"^\d+(?:-\d+)+$")


def _under_per_ring_corpus(path: Path) -> bool:
    """Return True if ``path`` falls under ``data/<tunnel_id>/r*/``.

    Per ``.cursorrules`` rule 3a, the per-ring corpora at
    ``data/<tunnel_id>/r*/**`` are read-only for everything except the
    explicit corpus regenerator. The v3 BO driver must never write there.
    """
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    parts = rel.parts
    if len(parts) < 2 or parts[0] != "data":
        return False
    return bool(_TUNNEL_ID_RE.match(parts[1]))


def assert_writable(target: Path | str) -> Path:
    """Raise ``ValueError`` if ``target`` is under any protected prefix.

    Returns the resolved ``Path`` so callers can chain.
    """
    p = Path(target)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    for root in PROTECTED_PREFIXES:
        if _is_within(p, root):
            raise ValueError(
                f"Refusing protected output path: {p} (under {root}). "
                "All v3 BO outputs must live under logs/v3/.../."
            )
    if _under_per_ring_corpus(p):
        raise ValueError(
            f"Refusing protected output path: {p} (under data/<tunnel>/r*/). "
            "All v3 BO outputs must live under logs/v3/.../."
        )
    return p


def assert_all_writable(paths: Iterable[Path | str]) -> None:
    for p in paths:
        assert_writable(p)
