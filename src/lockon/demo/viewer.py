"""viewer — Gradio Blocks over the pre-rendered shots + curves (SPEC.md `viewer.py`).

CLI: ``python -m lockon.demo.viewer [--check] [--port 7860]``

A tab per shot (`gr.Video` + a one-line caption with retention in the window, from
`demo/shots/manifest.json`), a "Curves" tab (`reports/curves/summary.png` +
`reports/sweep/results.json`) and a "Where it breaks" tab (`reports/sweep/failure_report.md`).
`--check` builds the Blocks object without launching, verifies every referenced file exists,
prints the manifest, and exits 0/1 (gate G7's second command) — that is the whole point of this
module being importable without ever importing `gradio`: `gradio` is only imported inside
`build_blocks`, so `import lockon.demo.viewer` stays cheap and the `--check` fallback (project.md
§4: mp4s + `demo/index.html` are enough) works even when `gradio` fails to import.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import gradio as gr

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = REPO_ROOT / "demo"
SHOTS_DIR = DEMO_DIR / "shots"
MANIFEST_PATH = SHOTS_DIR / "manifest.json"
INDEX_HTML = DEMO_DIR / "index.html"
REPORTS_DIR = REPO_ROOT / "reports"
CURVES_PNG = REPORTS_DIR / "curves" / "summary.png"
SWEEP_RESULTS = REPORTS_DIR / "sweep" / "results.json"
FAILURE_REPORT = REPORTS_DIR / "sweep" / "failure_report.md"


def _load_manifest() -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return data


def _curves_status() -> str:
    return "curves: present" if CURVES_PNG.exists() and SWEEP_RESULTS.exists() else "curves: pending"


def build_blocks() -> gr.Blocks:
    """Build (but do not launch) the Gradio Blocks. Only place `gradio` is imported."""
    import gradio as gr

    manifest = _load_manifest()
    with gr.Blocks(title="lockon demo") as demo:
        for name, entry in manifest.items():
            with gr.Tab(name):
                gr.Video(str(SHOTS_DIR / str(entry["file"])))
                retention_pct = cast(float, entry["retention"]) * 100.0
                gr.Markdown(f"{entry['caption']} — retention in window: {retention_pct:.0f}%")
        with gr.Tab("Curves"):
            if CURVES_PNG.exists():
                gr.Image(str(CURVES_PNG))
            else:
                gr.Markdown("curves: pending")
            if SWEEP_RESULTS.exists():
                results = json.loads(SWEEP_RESULTS.read_text(encoding="utf-8"))
                gr.Markdown(f"```json\n{json.dumps(results, indent=2)}\n```")
            else:
                gr.Markdown("curves: pending")
        with gr.Tab("Where it breaks"):
            if FAILURE_REPORT.exists():
                gr.Markdown(FAILURE_REPORT.read_text(encoding="utf-8"))
            else:
                gr.Markdown("curves: pending")
    return cast("gr.Blocks", demo)


def _missing_files() -> list[str]:
    """Every file `--check` requires to exist: the manifest, its referenced mp4s, and the static
    `index.html` fallback. `reports/curves/summary.png` and `reports/sweep/results.json` are
    deliberately excluded — the sweep runs later (SPEC.md) and their absence prints "curves:
    pending" instead of failing the check.
    """
    if not MANIFEST_PATH.exists():
        return [str(MANIFEST_PATH)]
    manifest = _load_manifest()
    missing = [str(SHOTS_DIR / str(entry["file"])) for entry in manifest.values() if not (SHOTS_DIR / str(entry["file"])).exists()]
    if not INDEX_HTML.exists():
        missing.append(str(INDEX_HTML))
    return missing


def check() -> int:
    missing = _missing_files()
    if missing:
        logger.error("viewer --check failed, missing: %s", missing)
        return 1

    try:
        build_blocks()
    except ImportError:
        logger.warning("gradio unavailable; falling back to the mp4-only check (project.md §4)")

    logger.info(_curves_status())
    # The manifest print is the CLI's actual output contract (SPEC.md), not a diagnostic log.
    print(json.dumps(_load_manifest(), indent=2))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="lockon demo viewer (Gradio).")
    p.add_argument("--check", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    if args.check:
        return check()
    demo = build_blocks()
    demo.launch(server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
