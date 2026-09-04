"""Package boundary guard (project.md §2, plan.md §9 D1). Cross-package import = gate failure.

Rule table (the only place it is written):
  core                      -> numpy, stdlib only
  env, sensor, track, policy -> lockon.core + third-party only
  harness                   -> anything (composition root)
  demo                      -> lockon.core, lockon.harness
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "lockon"
LIBRARY = ("env", "sensor", "track", "policy")
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    **{p: {"core"} for p in LIBRARY},
    "harness": {"core", "env", "sensor", "track", "policy", "harness"},
    "demo": {"core", "harness", "demo"},
}
CORE_THIRD_PARTY = {"numpy"}


def _lockon_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _package_files() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for pkg in ALLOWED:
        for f in (SRC / pkg).rglob("*.py"):
            out.append((pkg, f))
    return out


@pytest.mark.parametrize("pkg,path", _package_files(), ids=lambda x: str(x) if isinstance(x, Path) else x)
def test_no_cross_package_import(pkg: str, path: Path) -> None:
    for mod in _lockon_imports(path):
        if not mod.startswith("lockon"):
            continue
        parts = mod.split(".")
        target = parts[1] if len(parts) > 1 else ""
        assert target in ALLOWED[pkg] | {pkg}, f"{path.relative_to(SRC.parent)} imports {mod}; {pkg} may only import {sorted(ALLOWED[pkg])}"


@pytest.mark.parametrize("path", list((SRC / "core").rglob("*.py")), ids=lambda p: p.name)
def test_core_has_no_third_party_beyond_numpy(path: Path) -> None:
    import sys

    stdlib = set(sys.stdlib_module_names)
    for mod in _lockon_imports(path):
        root = mod.split(".")[0]
        assert root in stdlib or root in CORE_THIRD_PARTY or root == "lockon", f"core imports {mod}"


def test_all_packages_exist() -> None:
    for pkg in ALLOWED:
        assert (SRC / pkg / "__init__.py").exists(), f"missing package {pkg}"
