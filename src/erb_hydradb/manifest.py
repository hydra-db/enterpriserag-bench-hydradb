"""Run manifests and artifact checksums.

Every command that produces an artifact records what produced it: package
version, git commit of this repository, Python version, the run config, the
pinned upstream identifiers, timestamps, and sha256 of every file written.
``erb-hydradb verify`` recomputes the checksums.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, paths


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.repo_root(),
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(run_dir: str | Path, stage: str, config: dict | None, files: list[str | Path],
                   extra: dict | None = None) -> Path:
    """Append a stage record to ``<run_dir>/manifest.json`` and refresh SHA256SUMS."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mpath = run_dir / "manifest.json"
    manifest = {"stages": []}
    if mpath.exists():
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    record = {
        "stage": stage,
        "finished_at": now_iso(),
        "erb_hydradb_version": __version__,
        "repo_commit": git_commit(),
        "python": platform.python_version(),
        "upstream": {"erb_commit": paths.ERB_COMMIT, "questions_sha256": paths.QUESTIONS_SHA256},
        "config": config,
        "files": {str(Path(p).relative_to(run_dir)): sha256_file(p) for p in files if Path(p).exists()},
    }
    if extra:
        record.update(extra)
    manifest["stages"].append(record)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_sums(run_dir)
    return mpath


def write_sums(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{sha256_file(p)}  {p.relative_to(run_dir)}")
    out = run_dir / "SHA256SUMS"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_sums(run_dir: str | Path) -> list[str]:
    """Return a list of mismatched/missing files (empty means verified)."""
    run_dir = Path(run_dir)
    problems = []
    sums = run_dir / "SHA256SUMS"
    if not sums.exists():
        return ["SHA256SUMS missing"]
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        p = run_dir / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(p) != digest:
            problems.append(f"changed: {rel}")
    return problems
