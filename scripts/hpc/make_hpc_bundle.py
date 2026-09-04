"""PERUN transfer bundle: build / verify / unpack (plan.md §7 step 7).

Ported from ws/100_occlusion_mot/scripts/make_hpc_bundle.py and trimmed hard: lockon
ships no external datasets (plan.md §2 non-goal), so the bundle is just the repo
(`git archive HEAD`) plus an offline cp311 manylinux wheelhouse -- no nested payload
tars, no reid/detector/asset staging. The hash-manifest bundle, `_safe_extract`,
LF-only script emission (sweep_common.write_text_lf), and POSIX relpaths are kept
verbatim from the 100 pattern (CLAUDE.md: "100's bundle pattern verbatim").

    build   -- stage repo.tar (+ wheelhouse.tar unless --no-wheelhouse), hash, archive
    verify  -- re-check every manifest entry's sha256 (cluster side, stdlib only)
    unpack  -- expand into <dest>/repo (+ <dest>/wheelhouse)
    check-wheelhouse -- resolve requirements-hpc.txt offline against the wheelhouse

Usage (dev box):
  uv run python scripts/hpc/make_hpc_bundle.py build --no-wheelhouse   # stage only, no torch yet
  uv run python scripts/hpc/make_hpc_bundle.py build --refresh-wheelhouse   # real download, run when ready
  uv run python scripts/hpc/make_hpc_bundle.py check-wheelhouse

  # the exact wheelhouse download command (requirements-hpc.in header repeats it):
  uv run pip download -r requirements-hpc.txt -d data/wheelhouse --only-binary=:all: \\
      --platform manylinux_2_28_x86_64 --python-version 3.11 --implementation cp --abi cp311

Usage (cluster, system python3, stdlib only):
  python3 make_hpc_bundle.py verify
  python3 make_hpc_bundle.py unpack --dest .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                     stream=sys.stdout)
logger = logging.getLogger("make_hpc_bundle")

ROOT = Path(__file__).resolve().parents[2]

BUNDLE_VERSION = 1
BUNDLE_DIRNAME = "lockon_hpc"
MANIFEST_NAME = "MANIFEST.json"
DEFAULT_CONFIG = "configs/hpc_sweep.yaml"
LOCK_FILE = "requirements-hpc.txt"

GZIP_LEVEL = 1  # payload is mostly already-compressed wheels; higher levels buy little


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class Requirement:
    """One input the sweep needs on the cluster, beyond the repo itself."""

    key: str
    kind: str
    path: str  # repo-relative, POSIX
    reason: str


@dataclass
class EntrySpec:
    """One member of the bundle: a loose file, or a tree packed into a tar."""

    dest: str  # path inside the bundle
    kind: str  # "file" | "tar"
    sources: list[str] = field(default_factory=list)  # repo-relative, POSIX
    covers: list[str] = field(default_factory=list)  # Requirement.key values
    unpack_to: str = "repo"  # "repo" | "bundle" | "" (no unpack)
    note: str = ""


def required_inputs(root: Path = ROOT, include_wheelhouse: bool = True) -> list[Requirement]:
    """Every non-repo input the CPU sweep needs on the cluster. Derived, not hardcoded --
    the pytest completeness guard compares the bundle plan against this."""
    reqs = [
        Requirement("config:ppo_local", "repo_file", "configs/ppo_local.yaml",
                     "base config scripts/hpc/sweep_common.py::generate_unit_config derives "
                     "every per-unit YAML from"),
        Requirement("config:hpc_sweep", "repo_file", "configs/hpc_sweep.yaml",
                     "unit enumeration + budget config (single source of truth)"),
        Requirement("lock", "repo_file", LOCK_FILE,
                     "exact cluster environment, resolved for x86_64-manylinux_2_28"),
    ]
    if include_wheelhouse:
        reqs.append(Requirement(
            "wheelhouse", "wheelhouse", "data/wheelhouse",
            "offline pip install source: linux_x86_64 / cp311 wheels for requirements-hpc.txt",
        ))
    return reqs


def plan_entries(root: Path = ROOT, include_wheelhouse: bool = True) -> list[EntrySpec]:
    """The bundle layout. Every Requirement must be covered by exactly one entry."""
    entries = [
        EntrySpec(
            dest="repo.tar", kind="tar", sources=["@git-archive"],
            covers=["config:ppo_local", "config:hpc_sweep", "lock"], unpack_to="repo",
            note="git archive HEAD -- tracked files only, no .git, no gitignored junk",
        ),
    ]
    if include_wheelhouse:
        entries.append(EntrySpec(
            dest="wheelhouse.tar", kind="tar", sources=["data/wheelhouse"],
            covers=["wheelhouse"], unpack_to="bundle",
            note="unpacks beside repo/ as wheelhouse/, not into the repo tree",
        ))
    entries.append(EntrySpec(dest=MANIFEST_NAME, kind="file", sources=[], covers=[],
                              unpack_to=""))
    return entries


def coverage_gaps(root: Path = ROOT, include_wheelhouse: bool = True) -> list[str]:
    """Requirement keys the bundle plan does NOT ship. Empty list == complete."""
    covered = {k for e in plan_entries(root, include_wheelhouse) for k in e.covers}
    return sorted(r.key for r in required_inputs(root, include_wheelhouse)
                  if r.key not in covered)


# --------------------------------------------------------------------------- hashing


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- build


def git(*args: str, root: Path = ROOT) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def git_state(root: Path = ROOT) -> dict[str, Any]:
    try:
        dirty = git("status", "--porcelain", "--untracked-files=no", root=root)
        return {
            "commit": git("rev-parse", "HEAD", root=root),
            "short": git("rev-parse", "--short", "HEAD", root=root),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD", root=root),
            "dirty": bool(dirty),
            "dirty_paths": dirty.splitlines(),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # not a git repo
        logger.warning("git state unavailable: %s", exc)
        return {"commit": None, "short": "nogit", "branch": None, "dirty": False,
                "dirty_paths": []}


def _add_tree(tar: tarfile.TarFile, src: Path, arcname: str) -> tuple[int, int]:
    """Add a file or directory tree with POSIX arcnames. Returns (members, bytes)."""
    members = 0
    total = 0
    if src.is_file():
        info = tar.gettarinfo(str(src), arcname=arcname)
        info.mode = 0o755 if src.suffix in {".sh", ".sbatch"} else 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        with src.open("rb") as f:
            tar.addfile(info, f)
        return 1, info.size
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src).as_posix()
        info = tar.gettarinfo(str(path), arcname=f"{arcname}/{rel}")
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        with path.open("rb") as f:
            tar.addfile(info, f)
        members += 1
        total += info.size
    return members, total


def build_entry(spec: EntrySpec, stage: Path, root: Path) -> dict[str, Any]:
    dest = stage / spec.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    members = 0
    total = 0

    if spec.sources == ["@git-archive"]:
        logger.info("staging %s <- git archive HEAD", spec.dest)
        subprocess.run(["git", "archive", "--format=tar", "-o", str(dest), "HEAD"],
                       cwd=root, check=True)
        with tarfile.open(dest, "r:") as tar:
            infos = tar.getmembers()
        members = sum(1 for i in infos if i.isfile())
        total = sum(i.size for i in infos)
    else:
        missing = [s for s in spec.sources if not (root / s).exists()]
        if missing:
            raise FileNotFoundError(
                f"{spec.dest}: missing bundle input(s): {missing}\n"
                f"  (nothing is silently dropped -- stage the input or pass --no-wheelhouse)"
            )
        logger.info("staging %s <- %d source(s)", spec.dest, len(spec.sources))
        with tarfile.open(dest, "w:") as tar:
            for src in spec.sources:
                m, b = _add_tree(tar, root / src, Path(src).name)
                members += m
                total += b

    digest = sha256_file(dest)
    logger.info("  %s  %.3f GB  %d members  sha256=%s...",
                spec.dest, dest.stat().st_size / 1e9, members, digest[:16])
    return {
        "path": spec.dest,
        "kind": spec.kind,
        "sha256": digest,
        "bytes": dest.stat().st_size,
        "members": members,
        "member_bytes": total,
        "covers": spec.covers,
        "unpack_to": spec.unpack_to,
        "note": spec.note,
    }


def build(
    config: str = DEFAULT_CONFIG,
    root: Path = ROOT,
    out_dir: Path | None = None,
    archive: bool = True,
    allow_dirty: bool = False,
    include_wheelhouse: bool = True,
) -> Path:
    out_dir = out_dir or (root / "dist")

    state = git_state(root)
    if state["dirty"] and not allow_dirty:
        raise SystemExit(
            "refusing to build from a dirty worktree: repo.tar comes from `git archive "
            "HEAD`, so uncommitted changes would be SILENTLY ABSENT from the bundle.\n"
            "  commit them, or pass --allow-dirty if you truly mean to ship HEAD.\n"
            "  dirty: " + ", ".join(state["dirty_paths"][:10])
        )

    gaps = coverage_gaps(root, include_wheelhouse)
    if gaps:
        raise SystemExit(f"bundle plan does not cover required input(s): {gaps}")

    stage = out_dir / BUNDLE_DIRNAME
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    specs = [s for s in plan_entries(root, include_wheelhouse) if s.kind == "tar"]
    entries = [build_entry(s, stage, root) for s in specs]

    manifest: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": state,
        "config": config,
        "include_wheelhouse": include_wheelhouse,
        "requirements": [r.__dict__ for r in required_inputs(root, include_wheelhouse)],
        "entries": entries,
        "totals": {
            "bytes": sum(e["bytes"] for e in entries),
            "members": sum(e["members"] for e in entries),
        },
    }
    (stage / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), stage / "make_hpc_bundle.py")
    runbook = root / "HPC_RUNBOOK.md"
    if runbook.exists():
        shutil.copy2(runbook, stage / "HPC_RUNBOOK.md")

    if not include_wheelhouse:
        logger.warning(
            "--no-wheelhouse: bundle has NO offline install source. Populate "
            "data/wheelhouse (see requirements-hpc.in header) and rebuild before shipping."
        )

    logger.info("staged bundle: %s (%.2f GB, %d members)",
                stage, manifest["totals"]["bytes"] / 1e9, manifest["totals"]["members"])

    if not archive:
        return stage

    tgz = out_dir / f"{BUNDLE_DIRNAME}_{state['short']}.tar.gz"
    logger.info("compressing -> %s (gzip level %d)", tgz, GZIP_LEVEL)
    with tarfile.open(tgz, "w:gz", compresslevel=GZIP_LEVEL) as tar:
        tar.add(stage, arcname=BUNDLE_DIRNAME)
    digest = sha256_file(tgz)
    (tgz.parent / f"{tgz.name}.sha256").write_text(f"{digest}  {tgz.name}\n",
                                                     encoding="utf-8", newline="\n")
    shutil.copy2(stage / MANIFEST_NAME, tgz.parent / f"{tgz.name}.manifest.json")
    logger.info("bundle: %s  %.2f GB  sha256=%s", tgz, tgz.stat().st_size / 1e9, digest)
    return tgz


def linux_platform_tags() -> list[str]:
    """x86_64 manylinux platform tags a glibc-2.28 node can run. pip's --platform is
    LITERAL, so the floor tag alone silently drops real wheels shipped under an older
    tag (kept from ws/100's make_hpc_bundle.py -- same reasoning applies to torch's
    manylinux_2_28 floor)."""
    tags = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
    tags += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"]
    return tags


def refresh_lock(root: Path) -> None:
    cmd = [
        "uv", "pip", "compile", "requirements-hpc.in",
        "--python-platform", "x86_64-manylinux_2_28", "--python-version", "3.11",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu",
        "--index-strategy", "unsafe-best-match", "-o", LOCK_FILE,
    ]
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def refresh_wheelhouse(root: Path, python_exe: str) -> None:
    """Download linux_x86_64 / cp311 wheels for requirements-hpc.txt into
    data/wheelhouse. NOT run by default -- torch alone is ~200 MB even on the CPU
    index; `build` defaults to --no-wheelhouse until this has actually been run once."""
    dest = root / "data" / "wheelhouse"
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, "-m", "pip", "download", "-r", LOCK_FILE, "-d", str(dest),
           "--only-binary=:all:"]
    for tag in linux_platform_tags():
        cmd += ["--platform", tag]
    cmd += ["--python-version", "3.11", "--implementation", "cp", "--abi", "cp311",
            "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def check_wheelhouse(root: Path, python_exe: str) -> int:
    """Resolve requirements-hpc.txt offline against data/wheelhouse, for LINUX/cp311.
    As far as a Windows dev box can validate the cluster environment -- it cannot
    prove the wheels IMPORT on a real Linux node, only that they resolve."""
    wheelhouse = root / "data" / "wheelhouse"
    lock = root / LOCK_FILE
    if not lock.exists():
        logger.error("no %s -- run `build --refresh-lock` first", LOCK_FILE)
        return 2
    target = root / "dist" / "_resolve_probe"
    if target.exists():
        shutil.rmtree(target)
    cmd = [
        python_exe, "-m", "pip", "install", "--dry-run", "--ignore-installed",
        "--no-index", "--find-links", str(wheelhouse), "--only-binary=:all:",
        "--target", str(target), "--python-version", "3.11",
        "--implementation", "cp", "--abi", "cp311", "-r", LOCK_FILE,
    ]
    for tag in linux_platform_tags():
        cmd += ["--platform", tag]
    logger.info("resolving %s offline against %s (linux/cp311 target)", LOCK_FILE, wheelhouse)
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        logger.error("WHEELHOUSE INCOMPLETE:\n%s", (proc.stderr or proc.stdout)[-2000:])
        return 1
    n = len(list(wheelhouse.glob("*.whl")))
    size = sum(p.stat().st_size for p in wheelhouse.glob("*.whl"))
    logger.info("WHEELHOUSE OK: %d wheels, %.2f GB, every lock entry resolves offline",
                n, size / 1e9)
    return 0


# --------------------------------------------------------------------------- verify


def verify(bundle_dir: Path) -> int:
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.exists():
        logger.error("no %s in %s", MANIFEST_NAME, bundle_dir)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logger.info("bundle built %s from %s%s", manifest["created_utc"],
                manifest["git"]["short"], " (DIRTY)" if manifest["git"]["dirty"] else "")

    failures: list[str] = []
    for entry in manifest["entries"]:
        path = bundle_dir / entry["path"]
        if not path.exists():
            failures.append(f"{entry['path']}: MISSING")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            failures.append(f"{entry['path']}: size {size} != {entry['bytes']}")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            failures.append(
                f"{entry['path']}: sha256 {digest[:16]}... != {entry['sha256'][:16]}..."
            )
            continue
        logger.info("  OK  %-20s %8.3f GB  %7d members",
                    entry["path"], size / 1e9, entry["members"])

    if failures:
        for f in failures:
            logger.error("  FAIL %s", f)
        logger.error("VERIFY FAILED: %d/%d entries bad", len(failures), len(manifest["entries"]))
        return 1
    logger.info("VERIFY OK: %d entries, %.2f GB, %d members",
                len(manifest["entries"]), manifest["totals"]["bytes"] / 1e9,
                manifest["totals"]["members"])
    return 0


# --------------------------------------------------------------------------- unpack


def unpack(bundle_dir: Path, dest: Path) -> int:
    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    repo = dest / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    for entry in manifest["entries"]:
        target = {"repo": repo, "bundle": dest}.get(entry["unpack_to"])
        if target is None:
            continue
        src = bundle_dir / entry["path"]
        logger.info("unpacking %-20s -> %s", entry["path"], target)
        with tarfile.open(src, "r:*") as tar:
            _safe_extract(tar, target)

    logger.info("unpacked to %s (repo tree: %s)", dest, repo)
    return 0


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    """Extract, rejecting absolute paths and ../ escapes. Prefer tarfile's own "data"
    filter (3.12, backported to 3.11.4+); the manual scan is the fallback only."""
    resolved = target.resolve()
    if hasattr(tarfile, "data_filter"):
        tar.extractall(resolved, filter="data")
        return
    for member in tar.getmembers():
        out = (resolved / member.name).resolve()
        if not str(out).startswith(str(resolved)):
            raise RuntimeError(f"unsafe tar member escapes destination: {member.name}")
    tar.extractall(resolved)


# --------------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="stage + hash + archive the bundle (dev box)")
    b.add_argument("--config", default=DEFAULT_CONFIG)
    b.add_argument("--out-dir", type=Path, default=None)
    b.add_argument("--no-archive", action="store_true",
                   help="stop at the staged directory (transfer it with rsync instead)")
    b.add_argument("--allow-dirty", action="store_true",
                   help="build from HEAD even though the worktree has uncommitted changes")
    b.add_argument("--refresh-lock", action="store_true", help="recompile requirements-hpc.txt")
    b.add_argument("--refresh-wheelhouse", action="store_true",
                   help="re-download data/wheelhouse from the lock (multi-hundred-MB torch "
                        "download -- not run unless explicitly requested)")
    b.add_argument("--pip-python", default=sys.executable,
                   help="interpreter that owns pip for --refresh-wheelhouse")
    b.add_argument("--no-wheelhouse", action="store_true",
                   help="build without data/wheelhouse (stage/verify/unpack still work; "
                        "the resulting bundle has no offline install source)")

    v = sub.add_parser("verify", help="re-check every manifest sha256 (cluster side)")
    v.add_argument("--bundle-dir", type=Path, default=Path("."))

    u = sub.add_parser("unpack", help="expand staged tars into <dest>/repo (cluster side)")
    u.add_argument("--bundle-dir", type=Path, default=Path("."))
    u.add_argument("--dest", type=Path, default=Path("."))

    c = sub.add_parser("check-wheelhouse",
                        help="resolve the lock offline against the wheelhouse (dev box)")
    c.add_argument("--pip-python", default=sys.executable)

    args = ap.parse_args()

    if args.cmd == "check-wheelhouse":
        return check_wheelhouse(ROOT, args.pip_python)

    if args.cmd == "build":
        if args.refresh_lock:
            refresh_lock(ROOT)
        if args.refresh_wheelhouse:
            refresh_wheelhouse(ROOT, args.pip_python)
        build(config=args.config, out_dir=args.out_dir, archive=not args.no_archive,
              allow_dirty=args.allow_dirty, include_wheelhouse=not args.no_wheelhouse)
        return 0
    if args.cmd == "verify":
        return verify(args.bundle_dir)
    return unpack(args.bundle_dir, args.dest)


if __name__ == "__main__":
    sys.exit(main())
