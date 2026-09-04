# lockon.track — specification (lead-authored 2026-09-04; plan.md step 3, gate G3)

Import rule: `lockon.core`, `trackers` (Roboflow, Apache-2.0), `supervision` (its dependency,
MIT), `numpy`. No env, no sensor, no rendering. Input = per-step GT `Detection`s (one per alive
channel that sees the target, from `SensorFrame.gt` or straight from `WorldState.person_box`),
output = `Track`s + `LockStatus`. project.md §2: the tracker consumes GT boxes + injected noise.

## Public API
```python
@dataclass(frozen=True)
class NoiseConfig:
    miss_rate: float = 0.1          # P(drop a detection) per channel per step
    jitter_px: float = 4.0          # gaussian sigma on each box edge, pixels
    occlusion_dropout: float = 0.0  # extra P(drop) when the box is small (< 24 px tall) — partial occlusion proxy
    latency_steps: int = 0          # detections are delivered latency_steps late (FIFO)
    score_mean: float = 0.85; score_std: float = 0.1   # confidence noise, clipped to [0.05, 1]
    seed: int = 0
    @classmethod
    def from_dial(cls, level: float, seed: int = 0) -> NoiseConfig   # level in [0,1]: miss 0.02→0.3, jitter 2→10, occl 0→0.4, latency 0→2

class NoiseInjector:
    def __init__(self, cfg: NoiseConfig) -> None
    def __call__(self, gt: dict[str, Detection | None], t: int) -> list[Detection]
        # per channel: drop w.p. miss (+occlusion_dropout if small), jitter edges, resample score,
        # keep channel name; then push through the latency FIFO; returns what arrives at step t.
        # Deterministic given seed + call order.

class LockTracker:
    """ByteTrack over fused multi-channel detections. Perception is frozen during RL (project.md §7)."""
    def __init__(self, lost_track_buffer: int = 30, activation: float = 0.5,
                 high_conf: float = 0.6, min_iou: float = 0.1, min_consecutive: int = 1) -> None
    def reset(self) -> None
    def update(self, detections: list[Detection], t: int) -> tuple[list[Track], LockStatus]
        # 1. fuse: if several channels report at the same t, average their boxes (they all see the
        #    same person) and take the max score → at most one detection per step. Document why:
        #    ByteTrack would otherwise spawn one track per channel.
        # 2. sv.Detections(xyxy=(N,4), confidence=(N,)) → ByteTrackTracker.update → tracker_id.
        # 3. LockStatus: target_id fixed at the first step a track appears; locked iff a track with
        #    target_id exists at t; steps_since_lock counts consecutive unlocked steps (0 when locked).
    @property
    def target_id(self) -> int | None
```
`ByteTrackTracker(frame_rate=CONTROL_HZ, ...)` from `trackers`; call signature is
`update(sv.Detections) -> sv.Detections` with `.tracker_id`. When `confidence` is None all
detections are high-confidence — always pass confidence.

## Tests (`tests/test_track.py`) — synthetic, no MuJoCo; all deterministic
Scene generators (module-level helpers in the test file, also reusable by harness scenes):
- `walk_boxes(n, x0, dx)`: a 40×90 px box translating dx px/step; GT dict for rgb+depth+thermal.
- `occlusion_scene()`: 120 steps; box visible 0–39, GT None on ALL channels 40–59 (behind a
  pillar), visible again 60–119 continuing the same motion.
- `lights_cut_scene()`: 120 steps; rgb channel GT None from step 50 on (rgb dead / dark), depth
  and thermal keep reporting.
1. `test_noise_injector_deterministic_and_rates`: with miss_rate 0.3 over 2000 steps the drop
   fraction is within ±0.05; two injectors with equal seeds produce identical outputs.
2. `test_latency_delays_delivery`: latency 2 → nothing at t=0,1; detection from t=0 arrives at t=2.
3. `test_id_held_through_occlusion` (G3 GIF1 logic): occlusion_scene with
   `NoiseConfig.from_dial(0.3)`; `retention(gt_boxes, tracks)` ≥ 0.75 AND target_id unchanged
   across the gap (the box re-appears within lost_track_buffer=30 steps; the Kalman prediction
   must re-associate it). Assert also that steps 60–119 are ≥ 95 % retained.
4. `test_id_held_through_channel_dropout` (G3 GIF2 logic): lights_cut_scene, same noise;
   retention ≥ 0.9, one target_id throughout.
5. `test_lock_status_counts`: steps_since_lock increments while unlocked, resets to 0 on
   re-lock, `locked` False at steps with no track.
6. `test_fusion_yields_single_track`: three channels reporting the same box every step → exactly
   one track id ever appears.
7. `test_zero_noise_perfect`: NoiseConfig(miss_rate=0, jitter_px=0) on walk_boxes → retention 1.0.
Thresholds 0.75 / 0.9 are derived from the scene geometry (20-step gap < 30-step buffer with
noise level 0.3), not tuned; if they fail, diagnose the instrument (fusion, buffer, activation)
before touching them (plan.md §5 retry law; a change is a deviation-log row).

## Also deliver
- `lockon/track/__init__.py` exporting NoiseConfig, NoiseInjector, LockTracker.
- `mypy --strict src/lockon/track` clean (use `# type: ignore[import-untyped]` for trackers/
  supervision if they lack stubs, nothing broader).
- Do NOT render GIFs here — harness renders GIF1/GIF2 from these scenes at step 3b.
