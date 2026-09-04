# Third-party components — 110_lockon

Licence policy (project.md §10): Apache-2.0 / MIT / BSD only. No AGPL, no Ultralytics, no
external datasets. Every runtime dependency is listed; pins live in `requirements.txt`.

| component | licence | use |
|---|---|---|
| MuJoCo (`mujoco`) | Apache-2.0 | simulator, offscreen renderer |
| NumPy | BSD-3 | schemas, math |
| Gymnasium | MIT | env API |
| Stable-Baselines3 | MIT | PPO |
| PyTorch | BSD-3 | SB3 backend |
| Roboflow `trackers` | Apache-2.0 | ByteTrack |
| PyAV (`av`) | BSD-3 (bundled FFmpeg: LGPL-only build) | mp4 writing |
| imageio | BSD-2 | GIF writing |
| Pillow | MIT-CMU (HPND) | image ops |
| Gradio | Apache-2.0 | viewer |
| PyYAML | MIT | configs |
| matplotlib | PSF-based (BSD-compatible) | curves |
| pytest, mypy, ruff (dev) | MIT | tests, typing, lint |

No model weights, no datasets, no assets from third parties. The humanoid figure and arena are
hand-authored MJCF in this repo.
