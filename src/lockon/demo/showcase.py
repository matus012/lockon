"""showcase — the `demos/showcase/` reel (plan step 11 fix). RENDER ONLY.

CLI: ``python -m lockon.demo.showcase [--out demos/showcase] [--only NAME ...] [--fps 10]
[--steps N] [--list]``

Nothing here trains anything and nothing here produces a number that appears in README.md's
result tables. Every clip is one episode of policies that already exist (`models/*.zip`, the
scripted baselines), rendered through the same frozen perception stack every eval uses. The
retention counter burned into a clip is that single clip's own within-window retention — it is
NOT the n=80 held-out headline figure, and every clip says so on its face.

Sections (each `--only`-selectable):
  scenario_{open,dense,corridor,dark}   one hunter, one arena, 30 s
  split_{open,dense,corridor,dark}      static | scripted | PPO, same seed and clock
  pair_hunter                           untrained PPO vs trained PPO
  pair_evader                           random-walk evader vs trained evader
  evader_reel                           the learned evader, with per-step behaviour labels
  sensor_reel                           rgb | depth | thermal through a lights-cut, dropout flags
  sizzle                                the 60-90 s captioned cut of all of the above
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from lockon.core.schemas import ArenaLayout, Difficulty, WorldState
from lockon.harness.episode import EpisodeResult, resolve_hunter, run_episode
from lockon.harness.eval import resolve_prey
from lockon.harness.render import _write_gif, _write_mp4

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINED_HUNTER = "models/ppo_hunter.zip"
TRAINED_PREY = "models/learned_prey_seed0.zip"
UNTRAINED_HUNTER = "runs/showcase/untrained_hunter.zip"

CLIP_STEPS = 300  # 30 s at CONTROL_HZ = 10 — the brief's ">= 30 s each"
FPS = 10

BG = (14, 14, 16)
FG = (238, 238, 238)
DIM = (150, 150, 155)
LOCK_HELD = (60, 220, 90)  # the target's id, IoU >= 0.5 this step
LOCK_COAST = (250, 190, 40)  # the target's id alive but unmatched — the tracker is coasting
LOCK_OTHER = (235, 70, 70)  # any other id — an id switch is exactly what the metric counts as loss
PILLAR = (105, 105, 112)
DRONE = (255, 214, 0)
PERSON = (255, 70, 214)
LOS_OK = (60, 220, 90)
LOS_NO = (220, 70, 70)
MINIMAP = 150

FOOTER = "one episode, within-clip retention - not the n=80 held-out headline"


@dataclass(frozen=True)
class Scenario:
    name: str
    difficulty: Difficulty
    seed: int
    layout_style: str
    channel: str  # the panel shown in the single-view clips
    blurb: str


# The three geometry scenarios run with the lights ON (darkness 0.0). At the mid dial (0.5) the
# arena renders almost black and a viewer cannot see the occlusion the clip is about — the clip
# would be honest and useless. Every clip's HUD prints all five dials, so nothing is hidden; the
# README's numbers remain the mid-difficulty ones and are produced nowhere near this module.
SCENARIOS: dict[str, Scenario] = {
    "open": Scenario(
        name="open",
        difficulty=Difficulty(occluder_density=0.0, darkness=0.0),
        seed=101,
        layout_style="random",
        channel="rgb",
        blurb="Open arena, 4 pillars. Almost nothing to hide behind.",
    ),
    "dense": Scenario(
        name="dense",
        difficulty=Difficulty(occluder_density=1.0, darkness=0.0),
        seed=102,
        layout_style="random",
        channel="rgb",
        blurb="24 pillars. Cover everywhere - occlusion dominates the loss causes.",
    ),
    "corridor": Scenario(
        name="corridor",
        difficulty=Difficulty(occluder_density=0.5, darkness=0.0),
        seed=103,
        layout_style="corridor",
        channel="rgb",
        blurb="Two pillar rows, one lane. Repeated brief occlusions, fixed geometry.",
    ),
    "dark": Scenario(
        name="dark",
        difficulty=Difficulty(darkness=1.0),
        seed=104,
        layout_style="random",
        channel="thermal",
        blurb="Lights out. The colour channel is blind; the thermal proxy carries the lock.",
    ),
}

# --- showcase seed selection ------------------------------------------------------------------
# Showcase seeds are SELECTED, never tuned: the rule lives here, is published in this package's
# README, and is applied before any clip is rendered. `scenes.select_occlusion_seed` already works
# this way (deviation-log rows 7 and 14) and this is the same discipline over a longer clip.
#
# Why it is needed: over CLIP_STEPS (300, half again the standard episode) a scripted hunter that
# loses the target often never gets it back, and the metric's target id is dead for good once it
# switches. Seed 104's lights-out clip spends its last 170 steps as an entirely black frame, and
# seed 102's evader clip spends its last third pointed at a pillar. Both are true; neither shows
# a viewer anything. The criterion below is about STAGING — is the target in frame at all, and
# does something actually happen — and is computed from geometry (`person_visible`), never from
# retention, so it cannot select a seed that flatters a policy.
CORRIDOR_LANE_M = 3.0  # arena.CORRIDOR_X_M: |person.x| below this is inside the lane
CORRIDOR_MIN_LANE_FRACTION = 0.6
SHOWCASE_SEEDS = range(101, 161)
STAGE_MIN_VISIBLE = 0.45  # below this the clip is mostly an empty room
STAGE_MIN_GAP = 5  # and without one real occlusion it shows no failure either


def _longest_gap(visible: npt.NDArray[np.bool_]) -> int:
    best = run = 0
    for v in visible:
        run = 0 if v else run + 1
        best = max(best, run)
    return best


@lru_cache(maxsize=8)
def select_showcase_seed(name: str, prey: str = "scripted", steps: int = CLIP_STEPS) -> int:
    """First seed staging `name` well enough to be worth watching (see the note above)."""
    sc = SCENARIOS[name]
    for seed in SHOWCASE_SEEDS:
        res = run_episode(
            sc.difficulty,
            seed,
            resolve_hunter("scripted"),
            resolve_prey(prey),
            steps=steps,
            layout_style=sc.layout_style,
        )
        visible = np.array([st.person_visible for st in res.states], dtype=bool)
        frac, gap = float(visible.mean()), _longest_gap(visible)
        lane = float(np.mean([abs(st.person.x) < CORRIDOR_LANE_M for st in res.states]))
        ok = frac >= STAGE_MIN_VISIBLE and gap >= STAGE_MIN_GAP
        if sc.layout_style == "corridor":
            ok = ok and lane >= CORRIDOR_MIN_LANE_FRACTION
        logger.info(
            "%s seed %d: visible %.2f, longest gap %d, in-lane %.2f -> %s",
            name, seed, frac, gap, lane, "SELECTED" if ok else "no",
        )
        if ok:
            return seed
    raise RuntimeError(f"no seed stages {name!r} — report, do not tune")


@lru_cache(maxsize=8)
def resolve_scenario(name: str) -> Scenario:
    """`SCENARIOS[name]` with its showcase seed resolved by the staging rule on first use."""
    return Scenario(**{**vars(SCENARIOS[name]), "seed": select_showcase_seed(name)})


# ---------------------------------------------------------------------------------------------
# drawing primitives


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _blank(w: int, h: int, color: tuple[int, int, int] = BG) -> npt.NDArray[np.uint8]:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _text(
    panel: npt.NDArray[np.uint8],
    lines: list[tuple[str, tuple[int, int, int], int]],
    x: int,
    y: int,
    line_h: int = 22,
) -> npt.NDArray[np.uint8]:
    """Draw `(text, colour, size)` lines at `(x, y)`; returns a new array (PIL round trip)."""
    img = Image.fromarray(panel)
    draw = ImageDraw.Draw(img)
    for i, (s, color, size) in enumerate(lines):
        draw.text((x, y + i * line_h), s, fill=color, font=_font(size))
    return np.array(img, dtype=np.uint8)


def _rect(
    panel: npt.NDArray[np.uint8], box: npt.NDArray[np.float64], color: tuple[int, int, int], t: int = 3
) -> None:
    h, w = panel.shape[:2]
    x0, y0 = int(np.clip(box[0], 0, w - 1)), int(np.clip(box[1], 0, h - 1))
    x1, y1 = int(np.clip(box[2], 0, w - 1)), int(np.clip(box[3], 0, h - 1))
    panel[y0 : y0 + t, x0 : x1 + 1] = color
    panel[max(y1 - t + 1, 0) : y1 + 1, x0 : x1 + 1] = color
    panel[y0 : y1 + 1, x0 : x0 + t] = color
    panel[y0 : y1 + 1, max(x1 - t + 1, 0) : x1 + 1] = color


def _no_signal(w: int, h: int, t: int) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(t)
    panel = rng.integers(0, 40, size=(h, w, 3)).astype(np.uint8)
    return _text(panel, [("NO SIGNAL", (210, 70, 70), 26)], w // 2 - 62, h // 2 - 14)


def _channel_panel(result: EpisodeResult, i: int, channel: str) -> npt.NDArray[np.uint8]:
    assert result.frames is not None
    frame = result.frames[i]
    image = frame.images.get(channel)
    if image is None:
        return _no_signal(frame.width, frame.height, frame.t)
    panel = np.asarray(image)
    if panel.ndim == 2:
        panel = np.repeat(panel[:, :, None], 3, axis=2)
    return np.ascontiguousarray(panel[:, :, :3].astype(np.uint8))


def _overlay_tracks(panel: npt.NDArray[np.uint8], result: EpisodeResult, i: int) -> None:
    """Colour every track by what the metric makes of it: held / coasting / an id switch."""
    target = result.retention.target_id
    retained = bool(result.retention.retained[i])
    for tr in result.tracks[i]:
        if target is not None and tr.track_id == target:
            color = LOCK_HELD if retained else LOCK_COAST
        else:
            color = LOCK_OTHER
        _rect(panel, tr.box, color)


def _minimap(layout: ArenaLayout, state: WorldState, size: int = MINIMAP) -> npt.NDArray[np.uint8]:
    panel = _blank(size, size, (20, 20, 24))
    panel[0:2, :] = panel[-2:, :] = panel[:, 0:2] = panel[:, -2:] = (70, 70, 78)
    hs = layout.half_size

    def to_map(x: float, y: float) -> tuple[int, int]:
        u = (x / hs * 0.5 + 0.5) * size
        v = (1.0 - (y / hs * 0.5 + 0.5)) * size
        return int(np.clip(u, 0, size - 1)), int(np.clip(v, 0, size - 1))

    scale = size / (2.0 * hs)
    for px, py, hw in layout.pillars:
        cx, cy = to_map(float(px), float(py))
        r = max(1, int(hw * scale))
        panel[max(cy - r, 0) : cy + r + 1, max(cx - r, 0) : cx + r + 1] = PILLAR

    d = to_map(state.drone.x, state.drone.y)
    p = to_map(state.person.x, state.person.y)
    color = LOS_OK if state.person_visible else LOS_NO
    n = max(abs(p[0] - d[0]), abs(p[1] - d[1]), 1)
    for k in range(n + 1):
        x = round(d[0] + (p[0] - d[0]) * k / n)
        y = round(d[1] + (p[1] - d[1]) * k / n)
        if 0 <= x < size and 0 <= y < size:
            panel[y, x] = color
    for cx, cy, col in ((d[0], d[1], DRONE), (p[0], p[1], PERSON)):
        panel[max(cy - 3, 0) : cy + 4, max(cx - 3, 0) : cx + 4] = col
    return panel


def _fit(panel: npt.NDArray[np.uint8], w: int, h: int) -> npt.NDArray[np.uint8]:
    if panel.shape[1] == w and panel.shape[0] == h:
        return panel
    return np.array(Image.fromarray(panel).resize((w, h), Image.Resampling.LANCZOS), dtype=np.uint8)


def _hstack(panels: list[npt.NDArray[np.uint8]], gap: int = 6) -> npt.NDArray[np.uint8]:
    h = max(p.shape[0] for p in panels)
    out: list[npt.NDArray[np.uint8]] = []
    for k, p in enumerate(panels):
        if k:
            out.append(_blank(gap, h))
        pad = _blank(p.shape[1], h)
        pad[: p.shape[0]] = p
        out.append(pad)
    return np.concatenate(out, axis=1)


def _vstack(panels: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.uint8]:
    w = max(p.shape[1] for p in panels)
    out: list[npt.NDArray[np.uint8]] = []
    for p in panels:
        pad = _blank(w, p.shape[0])
        pad[:, : p.shape[1]] = p
        out.append(pad)
    return np.concatenate(out, axis=0)


def _pct(result: EpisodeResult, i: int) -> float:
    return float(result.retention.retained[: i + 1].mean()) * 100.0


def _lock_word(result: EpisodeResult, i: int) -> tuple[str, tuple[int, int, int]]:
    lock = result.lock[i]
    if bool(result.retention.retained[i]):
        return "LOCK HELD", LOCK_HELD
    if lock.locked:
        return "COASTING", LOCK_COAST
    return f"LOCK LOST ({lock.steps_since_lock})", LOCK_OTHER


def _titled_panel(
    result: EpisodeResult, i: int, channel: str, w: int, h: int, title: str
) -> npt.NDArray[np.uint8]:
    """One view: the channel image with tracker boxes, a title strip and a live lock/retention line."""
    panel = _channel_panel(result, i, channel)
    _overlay_tracks(panel, result, i)
    panel = _fit(panel, w, h)
    word, color = _lock_word(result, i)
    head = _text(_blank(w, 26), [(title, FG, 17)], 8, 4, line_h=0)
    foot = _blank(w, 26)
    foot = _text(foot, [(word, color, 17)], 8, 4, line_h=0)
    foot = _text(foot, [(f"retention {_pct(result, i):.0f} %", FG, 17)], w - 132, 4, line_h=0)
    return _vstack([head, panel, foot])


def _caption_card(w: int, h: int, lines: list[str], seconds: float = 1.6, fps: int = FPS) -> list[npt.NDArray[np.uint8]]:
    card = _blank(w, h)
    sizes = [26] + [19] * (len(lines) - 1)
    y = h // 2 - 14 * len(lines)
    card = _text(card, [(s, FG if k == 0 else DIM, sizes[k]) for k, s in enumerate(lines)], 48, y, line_h=32)
    return [card] * int(seconds * fps)


# ---------------------------------------------------------------------------------------------
# episodes


def _run(
    scenario: Scenario, hunter: str, prey: str, steps: int, seed: int | None = None
) -> EpisodeResult:
    logger.info("episode: %s hunter=%s prey=%s steps=%d", scenario.name, hunter, prey, steps)
    return run_episode(
        scenario.difficulty,
        seed if seed is not None else scenario.seed,
        resolve_hunter(hunter),
        resolve_prey(prey),
        steps=steps,
        capture=True,
        layout_style=scenario.layout_style,
    )


def _hunter_label(spec: str) -> str:
    if spec == "static":
        return "static camera"
    if spec == "scripted":
        return "scripted hunter"
    if UNTRAINED_HUNTER in spec:
        return "PPO hunter, UNTRAINED"
    return "PPO hunter, trained"


# ---------------------------------------------------------------------------------------------
# sections


def scenario_clip(scenario: Scenario, steps: int) -> list[npt.NDArray[np.uint8]]:
    """One 30 s look at a scenario under the hero (scripted) hunter."""
    result = _run(scenario, "scripted", "scripted", steps)
    d = scenario.difficulty
    dials = (
        f"occluders {d.occluder_density:.1f}  darkness {d.darkness:.1f}  "
        f"prey speed {d.prey_speed:.1f}  aggression {d.prey_aggressiveness:.1f}  "
        f"dropout {d.channel_dropout:.1f}"
    )
    frames: list[npt.NDArray[np.uint8]] = []
    for i in range(len(result.states)):
        view = _channel_panel(result, i, scenario.channel)
        _overlay_tracks(view, result, i)
        view = _fit(view, 960, 720)
        bar = _blank(960, MINIMAP + 12)
        word, color = _lock_word(result, i)
        bar = _text(
            bar,
            [
                (f"{scenario.name.upper()}  -  {scenario.blurb}", FG, 18),
                (dials, DIM, 15),
                (f"hunter: scripted   evader: scripted   seed {scenario.seed}   t={result.states[i].t}", DIM, 15),
                (f"{word}      retention so far {_pct(result, i):.0f} %", color, 20),
                (FOOTER, DIM, 14),
            ],
            12,
            8,
            line_h=27,
        )
        bar[6 : 6 + MINIMAP, 960 - MINIMAP - 10 : 960 - 10] = _minimap(result.layout, result.states[i])
        frames.append(_vstack([view, bar]))
    return frames


def split_clip(scenario: Scenario, steps: int) -> list[npt.NDArray[np.uint8]]:
    """static | scripted | PPO on the same seed, dials and clock."""
    specs = [("static", "static"), ("scripted", "scripted"), (TRAINED_HUNTER, "PPO (trained)")]
    results = [_run(scenario, spec, "scripted", steps) for spec, _ in specs]
    w, h = 470, 352
    frames: list[npt.NDArray[np.uint8]] = []
    for i in range(steps):
        row = _hstack(
            [_titled_panel(r, i, scenario.channel, w, h, label) for r, (_, label) in zip(results, specs)]
        )
        bar = _blank(row.shape[1], 116)
        bar = _text(
            bar,
            [
                (f"{scenario.name.upper()}  -  same seed ({scenario.seed}), same dials, same clock (t={i})", FG, 18),
                ("the evader reacts to whichever hunter it faces, so the three worlds diverge after t=0", DIM, 15),
                ("green = target id held   amber = tracker coasting   red = a different id (a lock loss)", DIM, 15),
                (FOOTER, DIM, 14),
            ],
            12,
            8,
            line_h=26,
        )
        frames.append(_vstack([row, bar]))
    return frames


def pair_clip(
    left: tuple[str, str, str], right: tuple[str, str, str], scenario: Scenario, steps: int, title: str, note: str
) -> list[npt.NDArray[np.uint8]]:
    """Two (hunter, prey, label) arms of the same scenario, side by side."""
    lr = _run(scenario, left[0], left[1], steps)
    rr = _run(scenario, right[0], right[1], steps)
    w, h = 640, 480
    frames: list[npt.NDArray[np.uint8]] = []
    for i in range(steps):
        row = _hstack(
            [
                _titled_panel(lr, i, scenario.channel, w, h, left[2]),
                _titled_panel(rr, i, scenario.channel, w, h, right[2]),
            ]
        )
        bar = _blank(row.shape[1], 112)
        bar = _text(
            bar,
            [
                (title, FG, 19),
                (f"{scenario.name} arena, seed {scenario.seed}, t={i}", DIM, 15),
                (note, DIM, 15),
                (FOOTER, DIM, 14),
            ],
            12,
            8,
            line_h=26,
        )
        frames.append(_vstack([row, bar]))
    return frames


BEHAVIOUR_DARK = 0.15  # rgb's min_illumination: below this the colour channel is blind
BEHAVIOUR_COVER_M = 2.5


def _behaviour(result: EpisodeResult, i: int) -> tuple[str, tuple[int, int, int]]:
    """A label for what the evader is doing, read off the ground-truth state — never a guess."""
    st = result.states[i]
    person = np.array([st.person.x, st.person.y])
    if not st.in_fov:
        return "OUT OF FRAME", LOCK_OTHER
    if not st.unoccluded:
        return "BREAKING LINE OF SIGHT", LOCK_OTHER
    if st.illumination_at_person < BEHAVIOUR_DARK:
        return "RUNNING INTO THE DARK", LOCK_COAST
    pillars = result.layout.pillars
    if pillars.shape[0]:
        near = float(np.min(np.linalg.norm(pillars[:, :2] - person, axis=1) - pillars[:, 2]))
        if near < BEHAVIOUR_COVER_M:
            return "HUGGING COVER", LOCK_COAST
    if i >= 5:
        prev = result.states[i - 5]
        d_now = float(np.hypot(st.person.x - st.drone.x, st.person.y - st.drone.y))
        d_prev = float(np.hypot(prev.person.x - prev.drone.x, prev.person.y - prev.drone.y))
        if d_now - d_prev > 0.5:
            return "OPENING THE GAP", LOCK_COAST
    return "IN THE OPEN", LOCK_HELD


def evader_reel(steps: int) -> list[npt.NDArray[np.uint8]]:
    """The trained evader alone, against the scripted hunter, with its behaviour labelled per step."""
    sc = SCENARIOS["dense"]
    prey_path = str(REPO_ROOT / TRAINED_PREY)
    # staged against THIS prey: the evader moves differently from the scripted one, so the dense
    # scenario's own seed does not necessarily stage a watchable episode for it
    seed = select_showcase_seed("dense", prey_path)
    result = _run(sc, "scripted", prey_path, steps, seed=seed)
    frames: list[npt.NDArray[np.uint8]] = []
    for i in range(steps):
        view = _channel_panel(result, i, "rgb")
        _overlay_tracks(view, result, i)
        view = _fit(view, 960, 720)
        label, color = _behaviour(result, i)
        bar = _blank(960, MINIMAP + 12)
        bar = _text(
            bar,
            [
                ("THE EVADER - a policy trained to break the lock", FG, 19),
                (f"behaviour: {label}", color, 22),
                (f"hunter: scripted   evader: models/learned_prey_seed0.zip   seed {seed}   t={i}", DIM, 15),
                (f"retention so far {_pct(result, i):.0f} %", FG, 17),
                ("labels are read from ground-truth state (fov / occlusion / light / range), not inferred", DIM, 14),
            ],
            12,
            8,
            line_h=27,
        )
        bar[6 : 6 + MINIMAP, 960 - MINIMAP - 10 : 960 - 10] = _minimap(result.layout, result.states[i])
        frames.append(_vstack([view, bar]))
    return frames


def sensor_reel(steps: int) -> list[npt.NDArray[np.uint8]]:
    """rgb | depth | thermal through a lights-cut, with per-channel alive/dead flags."""
    cut_at = steps // 3
    scenario = Scenario(
        name="sensor",
        difficulty=Difficulty(darkness=0.0, occluder_density=0.4, prey_speed=0.3, channel_dropout=0.8),
        seed=105,
        layout_style="random",
        channel="rgb",
        blurb="",
    )

    def scene(env, t: int) -> None:  # type: ignore[no-untyped-def]
        if t == cut_at:
            env.set_darkness(1.0)

    result = run_episode(
        scenario.difficulty,
        scenario.seed,
        resolve_hunter("scripted"),
        resolve_prey("scripted"),
        steps=steps,
        scene=scene,
        capture=True,
        layout_style=scenario.layout_style,
    )
    assert result.frames is not None
    w, h = 470, 352
    frames: list[npt.NDArray[np.uint8]] = []
    for i in range(steps):
        frame = result.frames[i]
        panels: list[npt.NDArray[np.uint8]] = []
        for c in ("rgb", "depth", "thermal"):
            panel = _channel_panel(result, i, c)
            _overlay_tracks(panel, result, i)
            panel = _fit(panel, w, h)
            alive = frame.alive.get(c, False)
            sees = result.states[i].channels_see.get(c, False)
            if alive:
                flag, fc = ("sees target" if sees else "alive, blind"), (LOCK_HELD if sees else LOCK_COAST)
            else:
                flag, fc = "DROPPED OUT", LOCK_OTHER
            head = _text(_blank(w, 26), [(c, FG, 18)], 8, 4, line_h=0)
            foot = _text(_blank(w, 26), [(flag, fc, 17)], 8, 4, line_h=0)
            panels.append(_vstack([head, panel, foot]))
        row = _hstack(panels)
        lit = "LIGHTS ON" if i < cut_at else "LIGHTS OUT"
        lit_color = FG if i < cut_at else LOCK_COAST
        bar = _blank(row.shape[1], 116)
        illum_line = (
            f"illumination at target {result.states[i].illumination_at_person:.2f}   "
            f"channel dropout dial 0.8   retention so far {_pct(result, i):.0f} %"
        )
        bar = _text(
            bar,
            [
                (f"SENSOR REEL  -  {lit}  (cut at t={cut_at})   t={i}", lit_color, 19),
                (illum_line, DIM, 15),
                ("depth and thermal have no light requirement; rgb needs 0.15 - that is why darkness alone", DIM, 15),
                ("never breaks the lock, and why a dropout burst does.", DIM, 15),
            ],
            12,
            8,
            line_h=26,
        )
        frames.append(_vstack([row, bar]))
    return frames


# ---------------------------------------------------------------------------------------------
# the cut


def sizzle(fps: int) -> list[npt.NDArray[np.uint8]]:
    """A 60-90 s captioned cut. Segments are re-rendered short, so this never holds every clip."""
    w = 1440
    seg: list[npt.NDArray[np.uint8]] = []

    def card(lines: list[str], sec: float = 1.8) -> None:
        seg.extend(_caption_card(w, 420, lines, sec, fps))

    def add(frames: list[npt.NDArray[np.uint8]], keep: int) -> None:
        step = max(1, len(frames) // keep)
        for f in frames[::step][:keep]:
            seg.append(_fit(f, w, int(f.shape[0] * w / f.shape[1])))

    card(
        [
            "lockon - keeping a camera locked on a moving person",
            "MuJoCo simulation - drone camera, three channels, a frozen ByteTrack lock",
            "metric: lock retention % = steps holding the target's FIRST track id at IoU >= 0.5",
        ],
        2.6,
    )
    card(["1. Four arenas", "open, dense, corridor, lights-out - the scripted hunter in each"])
    for name in ("open", "dense", "corridor", "dark"):
        add(scenario_clip(resolve_scenario(name), 110), 55)
    card(["2. Three hunters, one seed", "static camera | scripted hunter | PPO hunter"])
    add(split_clip(resolve_scenario("dense"), 130), 90)
    card(
        [
            "3. Before and after",
            "an untrained PPO hunter next to the trained one,",
            "and a random-walk evader next to one trained to break the lock",
        ]
    )
    add(
        pair_clip(
            (UNTRAINED_HUNTER, "scripted", "PPO hunter, UNTRAINED"),
            (TRAINED_HUNTER, "scripted", "PPO hunter, trained"),
            resolve_scenario("dense"),
            100,
            "before / after: the PPO hunter",
            "",
        ),
        55,
    )
    add(
        pair_clip(
            ("scripted", "random", "evader: random walk"),
            ("scripted", str(REPO_ROOT / TRAINED_PREY), "evader: trained to break the lock"),
            resolve_scenario("dense"),
            100,
            "before / after: the evader",
            "",
        ),
        55,
    )
    card(["4. What the evader does", "labels read from ground-truth state, not inferred"])
    add(evader_reel(110), 65)
    card(["5. The sensors", "rgb needs light. depth and thermal do not. any of them can drop out."])
    add(sensor_reel(120), 80)
    card(
        [
            "What actually happened",
            "no PPO policy beat the hand-written hunter - 5 local runs and a 45-unit sweep",
            "static 44.0 %   scripted 52.0 %   PPO 51.3 %   (n=80, held-out seeds)",
            "the negative result is the result. github.com/matus012/lockon",
        ],
        3.4,
    )
    return seg


# ---------------------------------------------------------------------------------------------
# driver


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _sections(steps: int, fps: int) -> dict[str, tuple[str, object]]:
    """name -> (caption, zero-arg builder). Lazy: nothing renders until a section is selected."""
    out: dict[str, tuple[str, object]] = {}
    for name, sc in SCENARIOS.items():
        out[f"scenario_{name}"] = (
            f"{sc.name} arena, scripted hunter: {sc.blurb}",
            lambda n=name: scenario_clip(resolve_scenario(n), steps),
        )
        out[f"split_{name}"] = (
            f"{sc.name} arena: static | scripted | PPO on one seed and clock",
            lambda n=name: split_clip(resolve_scenario(n), steps),
        )
    out["pair_hunter"] = (
        "before / after: an untrained PPO hunter beside the trained one",
        lambda: pair_clip(
            (UNTRAINED_HUNTER, "scripted", "PPO hunter, UNTRAINED"),
            (TRAINED_HUNTER, "scripted", "PPO hunter, trained"),
            resolve_scenario("dense"),
            steps,
            "before / after: the PPO hunter (same arena, same seed)",
            "over 80 held-out seeds at mid difficulty, training is worth +6.9 pts [+0.9, +13.0]",
        ),
    )
    out["pair_evader"] = (
        "before / after: a random-walk evader beside one trained to break the lock",
        lambda: pair_clip(
            ("scripted", "random", "evader: random walk"),
            ("scripted", str(REPO_ROOT / TRAINED_PREY), "evader: trained to break the lock"),
            resolve_scenario("dense"),
            steps,
            "before / after: the evader (same arena, same seed, scripted hunter both sides)",
            "it does seek cover - but over 80 held-out seeds it costs the scripted hunter only "
            "+2.1 pts vs a random walk [-6.7, +10.8]: no detectable difference",
        ),
    )
    out["evader_reel"] = ("the trained evader, behaviour labelled per step", lambda: evader_reel(steps))
    out["sensor_reel"] = ("rgb | depth | thermal through a lights-cut, with dropout flags", lambda: sensor_reel(steps))
    out["sizzle"] = ("the captioned cut of everything above", lambda: sizzle(fps))
    return out


GIF_TABLE = ("scenario_dense", "scenario_dark", "split_dense", "pair_evader", "evader_reel", "sensor_reel")
GIF_TARGET_BYTES = 4 * 1024 * 1024  # committed GIFs sit with the existing ones (1.9-4.8 MB)
# (frames, width) tried in order until one lands under the target. Frames are cut before width,
# because a three-panel split screen becomes unreadable long before it becomes smooth.
GIF_BUDGETS: tuple[tuple[int, int], ...] = ((70, 560), (48, 560), (36, 560), (28, 500), (22, 440), (16, 380))


def _gif_from(frames: list[npt.NDArray[np.uint8]], path: Path, fps: int) -> Path:
    """Write a README GIF under `GIF_TARGET_BYTES`, shrinking until it fits.

    A lit dense arena quantises to 19.7 MB at the first budget while the lights-out one is 0.6 MB,
    so a single fixed size either blows the 10 MB guard or throws away the clips that were cheap.
    """
    for n_frames, width in GIF_BUDGETS:
        step = max(1, len(frames) // n_frames)
        picked = frames[::step][:n_frames]
        scaled = [_fit(f, width, int(f.shape[0] * width / f.shape[1])) for f in picked]
        _write_gif(scaled, path, max(4, fps // 2))
        size = path.stat().st_size
        if size <= GIF_TARGET_BYTES:
            logger.info("gif %s: %d frames @ %d px -> %.2f MB", path.name, len(scaled), width, size / 1e6)
            return path
        logger.info("gif %s: %.2f MB at (%d, %d), shrinking", path.name, size / 1e6, n_frames, width)
    logger.warning("gif %s still %.2f MB at the smallest budget", path.name, path.stat().st_size / 1e6)
    return path


def _write_gif_for(name: str, frames: list[npt.NDArray[np.uint8]], fps: int) -> Path:
    return _gif_from(frames, REPO_ROOT / "reports" / "gifs" / f"showcase_{name}.gif", fps)


def _gifs_from_mp4(out: Path, names: list[str], fps: int) -> dict[str, dict[str, object]]:
    """Rebuild the README GIFs by reading back the already-rendered mp4s — no re-simulation."""
    import av

    built: dict[str, dict[str, object]] = {}
    for name in names:
        mp4 = out / f"{name}.mp4"
        if not mp4.exists():
            logger.warning("no mp4 for %s; skipping its gif", name)
            continue
        with av.open(str(mp4)) as container:
            frames = [
                np.asarray(f.to_ndarray(format="rgb24"), dtype=np.uint8)
                for f in container.decode(video=0)
            ]
        gif = _write_gif_for(name, frames, fps)
        built[name] = {
            "gif": gif.relative_to(REPO_ROOT).as_posix(),
            "gif_mb": round(gif.stat().st_size / 1e6, 2),
        }
    return built


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Render the demos/showcase reel (render only, no training).")
    p.add_argument("--out", type=Path, default=Path("demos/showcase"))
    p.add_argument("--only", nargs="*", default=None, help="section names; default = all")
    p.add_argument("--steps", type=int, default=CLIP_STEPS)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--no-gifs", action="store_true")
    p.add_argument("--gifs-only", action="store_true",
                   help="rebuild the README GIFs from the existing mp4s; render nothing")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    sections = _sections(args.steps, args.fps)
    if args.list:
        for name, (caption, _) in sections.items():
            print(f"{name:20s} {caption}")
        return 0

    names = args.only if args.only else list(sections)
    unknown = [n for n in names if n not in sections]
    if unknown:
        p.error(f"unknown section(s): {unknown}; --list to see them")

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.gifs_only:
        for name, gif_entry in _gifs_from_mp4(out, [n for n in names if n in GIF_TABLE], args.fps).items():
            manifest.setdefault(name, {}).update(gif_entry)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({k: manifest[k] for k in manifest if k in GIF_TABLE}, indent=2))
        return 0

    for name in names:
        caption, build = sections[name]
        frames = build()  # type: ignore[operator]
        mp4 = out / f"{name}.mp4"
        _write_mp4(frames, mp4, args.fps)
        entry: dict[str, object] = {
            "caption": caption,
            "file": mp4.name,
            "frames": len(frames),
            "fps": args.fps,
            "seconds": round(len(frames) / args.fps, 1),
            "size_mb": round(mp4.stat().st_size / 1e6, 2),
            "git_head": _git_head(),
        }
        if not args.no_gifs and name in GIF_TABLE:
            gif = _write_gif_for(name, frames, args.fps)
            entry["gif"] = gif.relative_to(REPO_ROOT).as_posix()
            entry["gif_mb"] = round(gif.stat().st_size / 1e6, 2)
        manifest[name] = entry
        logger.info("%s: %.1f s, %.2f MB", name, entry["seconds"], entry["size_mb"])
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({k: manifest[k] for k in names}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
