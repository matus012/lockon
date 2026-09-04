"""Render harness — the one place every visual the owner sees is produced (SPEC.md `render.py`).

CLI: ``python -m lockon.harness.render --scene S --gif PATH [--mp4 PATH]
[--policy static|scripted|<zip>] [--fps 10] [--steps N] [--start T] [--scale 1.0]``

Runs the scene with ``capture=True``, composes per step: ``sensor.side_by_side`` on top, tracker
boxes + ids on every alive channel panel, a banner line, and a top-down minimap. Writes a GIF
(imageio) and optionally an MP4 (PyAV, h264/yuv420p). ``--start``/``--steps`` trim a long scene to
its interesting window for rendering while the underlying episode still runs to full length, so a
scripted event timed against the absolute step count (e.g. `lights_cut` at t=60) still fires.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import av
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from lockon.core.schemas import CHANNELS, ArenaLayout, LockStatus, SensorFrame, Track, WorldState
from lockon.env.env import Env
from lockon.harness.episode import EpisodeResult, resolve_hunter, run_episode
from lockon.harness.scenes import SCENE_WINDOWS, SCENES, SceneSpec
from lockon.sensor.sensor import side_by_side

logger = logging.getLogger(__name__)

MINIMAP_SIZE = 160
TRACK_BOX_COLOR = (0, 255, 255)
TRACK_BOX_THICKNESS = 2
LOS_VISIBLE_COLOR = (60, 220, 60)
LOS_HIDDEN_COLOR = (220, 60, 60)
DRONE_COLOR = (255, 220, 0)
PERSON_COLOR = (255, 60, 220)
PILLAR_COLOR = (110, 110, 110)
ARENA_BG_COLOR = (18, 18, 18)
ARENA_LINE_COLOR = (90, 90, 90)
BANNER_TEXT_COLOR = (255, 255, 255)
MP4_CRF = 23


def _draw_track_box(panel: npt.NDArray[np.uint8], box: npt.NDArray[np.float64]) -> None:
    """Numpy rectangle outline for one tracker box, clipped to `panel`'s bounds."""
    h, w = panel.shape[:2]
    x0, y0, x1, y1 = box
    xi0, yi0 = int(np.clip(x0, 0, w - 1)), int(np.clip(y0, 0, h - 1))
    xi1, yi1 = int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1))
    t = TRACK_BOX_THICKNESS
    panel[yi0 : yi0 + t, xi0 : xi1 + 1] = TRACK_BOX_COLOR
    panel[max(yi1 - t + 1, 0) : yi1 + 1, xi0 : xi1 + 1] = TRACK_BOX_COLOR
    panel[yi0 : yi1 + 1, xi0 : xi0 + t] = TRACK_BOX_COLOR
    panel[yi0 : yi1 + 1, max(xi1 - t + 1, 0) : xi1 + 1] = TRACK_BOX_COLOR


def _draw_tracks(top: npt.NDArray[np.uint8], frame_width: int, tracks: list[Track]) -> None:
    """Draw every track's box on each alive channel panel of the `side_by_side` strip."""
    for i, c in enumerate(CHANNELS):
        for tr in tracks:
            box = tr.box.copy()
            box[[0, 2]] += i * frame_width
            _draw_track_box(top, box)


def _world_to_map(x: float, y: float, half_size: float) -> tuple[int, int]:
    u = (x / half_size * 0.5 + 0.5) * MINIMAP_SIZE
    v = (1.0 - (y / half_size * 0.5 + 0.5)) * MINIMAP_SIZE
    return int(np.clip(u, 0, MINIMAP_SIZE - 1)), int(np.clip(v, 0, MINIMAP_SIZE - 1))


def _draw_square(panel: npt.NDArray[np.uint8], cx: int, cy: int, half: int, color: tuple[int, int, int]) -> None:
    h, w = panel.shape[:2]
    x0, x1 = int(np.clip(cx - half, 0, w - 1)), int(np.clip(cx + half, 0, w - 1))
    y0, y1 = int(np.clip(cy - half, 0, h - 1)), int(np.clip(cy + half, 0, h - 1))
    panel[y0 : y1 + 1, x0 : x1 + 1] = color


def _draw_line(panel: npt.NDArray[np.uint8], p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int]) -> None:
    h, w = panel.shape[:2]
    n = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]), 1)
    for k in range(n + 1):
        x = round(p0[0] + (p1[0] - p0[0]) * k / n)
        y = round(p0[1] + (p1[1] - p0[1]) * k / n)
        if 0 <= x < w and 0 <= y < h:
            panel[y, x] = color


def _minimap(layout: ArenaLayout, state: WorldState, visible: bool) -> npt.NDArray[np.uint8]:
    """160-px top-down minimap: arena square, pillars, drone triangle w/ heading, person dot, LOS line."""
    panel = np.full((MINIMAP_SIZE, MINIMAP_SIZE, 3), ARENA_BG_COLOR, dtype=np.uint8)
    panel[0:2, :] = ARENA_LINE_COLOR
    panel[-2:, :] = ARENA_LINE_COLOR
    panel[:, 0:2] = ARENA_LINE_COLOR
    panel[:, -2:] = ARENA_LINE_COLOR

    scale = MINIMAP_SIZE / (2.0 * layout.half_size)
    for px, py, hw in layout.pillars:
        cx, cy = _world_to_map(float(px), float(py), layout.half_size)
        _draw_square(panel, cx, cy, max(1, int(hw * scale)), PILLAR_COLOR)

    drone_xy = _world_to_map(state.drone.x, state.drone.y, layout.half_size)
    person_xy = _world_to_map(state.person.x, state.person.y, layout.half_size)
    los_color = LOS_VISIBLE_COLOR if visible else LOS_HIDDEN_COLOR
    _draw_line(panel, drone_xy, person_xy, los_color)

    hx = drone_xy[0] + int(6 * np.cos(-state.drone.yaw))
    hy = drone_xy[1] + int(6 * np.sin(-state.drone.yaw))
    _draw_line(panel, drone_xy, (hx, hy), DRONE_COLOR)
    _draw_square(panel, drone_xy[0], drone_xy[1], 3, DRONE_COLOR)
    _draw_square(panel, person_xy[0], person_xy[1], 3, PERSON_COLOR)
    return panel


def _banner_text(state: WorldState, lock: LockStatus, retention_so_far: float) -> str:
    lock_txt = "HELD" if lock.locked else f"LOST ({lock.steps_since_lock} steps)"
    return f"t={state.t} | lock: {lock_txt} | retention so far {retention_so_far * 100.0:.0f} %"


def _panel_legend(frame: SensorFrame) -> str:
    dead = [c for c, alive in frame.alive.items() if not alive]
    return "channels: " + " | ".join(frame.alive) + (f"   dead: {', '.join(dead)}" if dead else "")


def _compose_frame(
    result: EpisodeResult,
    layout: ArenaLayout,
    i: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    scale: float = 1.0,
) -> npt.NDArray[np.uint8]:
    """One rendered step: side-by-side channels + track boxes on top, banner + minimap strip below.

    `scale` shrinks the channel panels only; banner text and minimap stay full size so a 0.35x
    GIF is still legible (visual review 2026-09-04).
    """
    assert result.frames is not None
    frame = result.frames[i]
    top = side_by_side(frame, labels=True)
    _draw_tracks(top, frame.width, result.tracks[i])
    if scale != 1.0:
        img_top = Image.fromarray(top)
        new_size = (max(1, int(img_top.width * scale)), max(1, int(img_top.height * scale)))
        top = np.asarray(img_top.resize(new_size, Image.Resampling.LANCZOS), dtype=np.uint8)

    strip_h = MINIMAP_SIZE
    strip_w = top.shape[1]
    canvas = np.full((top.shape[0] + strip_h, strip_w, 3), ARENA_BG_COLOR, dtype=np.uint8)
    canvas[: top.shape[0]] = top

    minimap = _minimap(layout, result.states[i], result.states[i].person_visible)
    canvas[top.shape[0] :, strip_w - MINIMAP_SIZE :] = minimap

    retained = result.retention.retained
    retention_so_far = float(retained[: i + 1].mean()) if i + 1 <= len(retained) else 0.0
    text = _banner_text(result.states[i], result.lock[i], retention_so_far)

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    text_y = top.shape[0] + strip_h // 2 - 10
    draw.text((12, text_y), text, fill=BANNER_TEXT_COLOR, font=font)
    draw.text((12, text_y - 24), _panel_legend(frame), fill=BANNER_TEXT_COLOR, font=font)
    return np.asarray(img, dtype=np.uint8)


def render_frames(
    scene_name: str,
    *,
    policy: str | None = None,
    fps: int = 10,
    steps: int | None = None,
    start: int | None = None,
    scale: float = 1.0,
) -> list[npt.NDArray[np.uint8]]:
    """Run `scene_name` (SCENES registry) with capture=True and compose every rendered step.

    `steps`/`start` trim the *rendered window*: when `start` is given the underlying episode
    still runs its full scripted length (`spec.steps`) so a timed scripted event still fires;
    `steps` alone (no `start`) instead shortens the run itself (cheap smoke runs).
    """
    spec: SceneSpec = SCENES[scene_name]
    hunter = resolve_hunter(policy) if policy is not None else spec.hunter
    run_steps = steps if (start is None and steps is not None) else spec.steps

    result = run_episode(
        spec.difficulty, spec.seed, hunter, spec.prey, steps=run_steps, scene=spec.scene, capture=True
    )
    layout = Env(spec.difficulty, spec.seed).layout

    window_start = start or 0
    window_len = steps if (start is not None and steps is not None) else run_steps - window_start
    window_end = min(run_steps, window_start + window_len)

    font = ImageFont.load_default(size=16)
    return [_compose_frame(result, layout, i, font, scale) for i in range(window_start, window_end)]


def _write_gif(frames: list[npt.NDArray[np.uint8]], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, extension=".gif", duration=1000.0 / fps, loop=0)
    logger.info("wrote GIF %s (%d frames, %.1f fps)", path, len(frames), fps)


def _write_mp4(frames: list[npt.NDArray[np.uint8]], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    w_even, h_even = w - (w % 2), h - (h % 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w_even
    stream.height = h_even
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(MP4_CRF)}
    for f in frames:
        vframe = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h_even, :w_even]), format="rgb24")
        for packet in stream.encode(vframe):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    logger.info("wrote MP4 %s (%d frames, %d fps)", path, len(frames), fps)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a lockon.harness scene to GIF/MP4.")
    p.add_argument("--scene", required=True, choices=sorted(SCENES))
    p.add_argument("--gif", required=True, type=Path)
    p.add_argument("--mp4", type=Path, default=None)
    p.add_argument("--policy", default=None, help="static|scripted|<path to a PPOHunter zip>")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--start", type=str, default=None, help="window start step, or 'auto' = the scene's own window (SCENE_WINDOWS)")
    p.add_argument("--scale", type=float, default=1.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    start: int | None
    steps: int | None = args.steps
    if args.start == "auto":
        start, steps = SCENE_WINDOWS[args.scene]()
    else:
        start = int(args.start) if args.start is not None else None
    frames = render_frames(
        args.scene, policy=args.policy, fps=args.fps, steps=steps, start=start, scale=args.scale
    )
    _write_gif(frames, args.gif, args.fps)
    if args.mp4 is not None:
        _write_mp4(frames, args.mp4, args.fps)


if __name__ == "__main__":
    main()
