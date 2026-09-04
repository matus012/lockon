"""render_all — the three showcase shots + manifest.json (SPEC.md `render_all.py`).

CLI: ``python -m lockon.demo.render_all [--policy best|scripted] [--out demo/shots] [--steps N]
[--fps FPS]``

Renders the three pinned shots (110_plain.md: occlusion, lights_cut, chase) via
`lockon.harness.render.render_frames`, writes each mp4, a `manifest.json` (scene, seed, policy,
window, retention in the window, git HEAD) and a static `index.html` next to the shots (the
mp4-only Gradio fallback, project.md §4). Idempotent; exits 1 if any mp4 is missing or < 10 KB
afterwards.

`--steps N` overrides every shot's window to a plain `N`-step run (`start=None`) — the tiny,
fast path used by the render test and by ad-hoc smoke checks; without it the pinned windows
(occlusion 118..178, lights_cut 46..100, chase full 200) are used.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lockon.harness.episode import EpisodeResult, resolve_hunter, run_episode
from lockon.harness.render import _write_mp4, render_frames
from lockon.harness.scenes import SCENES, SceneSpec

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MIN_MP4_BYTES = 10_000


@dataclass(frozen=True)
class ShotSpec:
    scene: str
    start: int | None
    steps: int | None
    hero: bool  # True: hunter follows --policy (best.zip else scripted); False: scene's own hunter
    caption: str


SHOTS: dict[str, ShotSpec] = {
    "occlusion": ShotSpec(
        scene="occlusion",
        start=118,
        steps=60,
        hero=True,
        caption="Person ducks behind a pillar, drone repositions, lock survives.",
    ),
    "lights_cut": ShotSpec(
        scene="lights_cut",
        start=46,
        steps=54,
        hero=False,
        caption="Lights cut: color view goes black, heat view takes over, lock survives.",
    ),
    "chase": ShotSpec(
        scene="chase",
        start=None,
        steps=None,
        hero=True,
        caption="The chase: hero hunter vs prey at mid difficulty, full run.",
    ),
}


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _resolve_policy(shot: ShotSpec, policy_flag: str) -> str | None:
    """None means "use the scene's own hunter" (lights_cut's fixed StaticCamera)."""
    if not shot.hero:
        return None
    if policy_flag == "best":
        best = REPO_ROOT / "runs" / "ppo_local" / "best.zip"
        if best.exists():
            return str(best)
    return "scripted"


def _window(run_steps: int, start: int | None, steps: int | None) -> tuple[int, int]:
    """Mirrors `render.render_frames`'s window arithmetic (kept in sync, see its docstring)."""
    window_start = start or 0
    window_len = steps if (start is not None and steps is not None) else run_steps - window_start
    window_end = min(run_steps, window_start + window_len)
    return window_start, window_end


def _shot_retention(
    spec: SceneSpec, policy: str | None, start: int | None, steps: int | None
) -> tuple[float, str]:
    """Independent (capture=False) episode run, same seed/hunter/window as the render — cheap,
    since it skips the sensor pass — used only to report retention in the manifest.
    """
    hunter = resolve_hunter(policy) if policy is not None else spec.hunter
    run_steps = steps if (start is None and steps is not None) else spec.steps
    result: EpisodeResult = run_episode(
        spec.difficulty, spec.seed, hunter, spec.prey, steps=run_steps, scene=spec.scene
    )
    w0, w1 = _window(run_steps, start, steps)
    retained = result.retention.retained[w0:w1]
    retention_frac = float(retained.mean()) if retained.size else 0.0
    return retention_frac, hunter.name


def _render_one(
    name: str, shot: ShotSpec, policy_flag: str, out_dir: Path, fps: int, steps_override: int | None
) -> dict[str, object]:
    start = None if steps_override is not None else shot.start
    steps = steps_override if steps_override is not None else shot.steps
    policy = _resolve_policy(shot, policy_flag)

    spec = SCENES[shot.scene]
    retention_frac, policy_name = _shot_retention(spec, policy, start, steps)
    run_steps = steps if (start is None and steps is not None) else spec.steps
    w0, w1 = _window(run_steps, start, steps)

    frames = render_frames(shot.scene, policy=policy, fps=fps, steps=steps, start=start)
    mp4_path = out_dir / f"{name}.mp4"
    _write_mp4(frames, mp4_path, fps)
    logger.info("shot %s: %d frames -> %s", name, len(frames), mp4_path)

    return {
        "scene": shot.scene,
        "seed": spec.seed,
        "policy": policy_name,
        "window": {"start": w0, "steps": w1 - w0},
        "retention": retention_frac,
        "caption": shot.caption,
        "file": mp4_path.name,
        "git_head": _git_head(),
    }


def _write_index_html(out_dir: Path, manifest: dict[str, dict[str, object]]) -> Path:
    """Static `<video controls>` page, the mp4-only fallback (project.md §4) when Gradio is
    unavailable. Written next to `out_dir` (`out_dir.parent / "index.html"`) so it sits at
    `demo/index.html` for the real run and inside the test's tmp dir for the test run.
    """
    rows = []
    for name, entry in manifest.items():
        src = f"{out_dir.name}/{entry['file']}"
        retention_pct = cast(float, entry["retention"]) * 100.0
        rows.append(
            "<section>"
            f"<h2>{name}</h2>"
            f'<video controls preload="metadata" src="{src}"></video>'
            f"<p>{entry['caption']} (retention in window: {retention_pct:.0f}%)</p>"
            "</section>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>lockon demo</title></head>"
        "<body>" + "".join(rows) + "</body></html>"
    )
    index_path = out_dir.parent / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def render_all(
    policy: str, out: Path, steps: int | None = None, fps: int = 10
) -> dict[str, dict[str, object]]:
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for name, shot in SHOTS.items():
        manifest[name] = _render_one(name, shot, policy, out, fps, steps)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_index_html(out, manifest)
    return manifest


def _shots_ok(out: Path) -> bool:
    ok = True
    for name in SHOTS:
        p = out / f"{name}.mp4"
        if not p.exists() or p.stat().st_size < MIN_MP4_BYTES:
            logger.error("shot missing or too small (< %d bytes): %s", MIN_MP4_BYTES, p)
            ok = False
    return ok


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render the three lockon demo shots + manifest.json.")
    p.add_argument("--policy", choices=["best", "scripted"], default="best")
    p.add_argument("--out", type=Path, default=Path("demo/shots"))
    p.add_argument("--steps", type=int, default=None, help="override: shrink every shot to N steps")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    manifest = render_all(args.policy, args.out, args.steps, args.fps)
    if not _shots_ok(args.out):
        return 1
    logger.info("manifest:\n%s", json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
