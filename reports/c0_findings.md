# C0 findings — 110_lockon

date: 2026-09-04
searched by: 2 subagents (sonnet) — PyPI, GitHub, project docs, MuJoCo docs, SB3 docs

## Where I searched
PyPI (versions + licence metadata) · GitHub READMEs and release notes (supervision, trackers,
boxmot, bytetracker, motpy, mujoco, stable-baselines3, gymnasium, cleanrl, gradio, PyAV,
imageio-ffmpeg) · MuJoCo python docs (Renderer, MUJOCO_GL) · SB3 docs (VecFrameStack, save/load)
· ws/100_occlusion_mot (prior internal work: HPC bundle + license-guard pattern).

## A0 — access gate
No datasets in this project (project.md §2). All candidates are pip-installable, no
registration. PASS by construction.

## Candidates
| # | candidate | licence | fitness | verdict |
|---|---|---|---|---|
| 1 | `trackers` (Roboflow) 2.6 | Apache-2.0 | ByteTrack, SORT, OC-SORT, BoT-SORT; detector-agnostic boxes in | **adopt** (primary tracker) |
| 2 | `supervision` 0.28 ByteTrack | MIT | present but deprecated, removal at 0.31 | **reject**: sunset path |
| 3 | `bytetracker` (kadirnar) 0.3.2 | MIT | small, unmaintained since 2023 | **adapt**: vendor source as fallback only |
| 4 | `boxmot` 19 | AGPL-3.0 | — | **reject**: licence |
| 5 | `motpy` 0.0.10 | MIT | dead since 2021 | **reject**: unmaintained |
| 6 | `mujoco` 3.12 | Apache-2.0 | Renderer rgb/depth/segmentation; GLFW→WGL offscreen on Windows without a visible window | **adopt** |
| 7 | `dm_control` humanoid.xml | Apache-2.0 | source of a ready humanoid MJCF | **reject**: extra dep for one XML; hand-author capsule figure (project.md §3) |
| 8 | `stable-baselines3` 2.9 | MIT | PPO, VecFrameStack on vector obs, save/load + `reset_num_timesteps=False` resume; needs gymnasium ≥0.29.1 <2, torch ≥2.8 | **adopt** |
| 9 | `gymnasium` 1.3 | MIT | env API | **adopt** |
| 10 | `cleanrl` | MIT | single-file scripts, not a library | **adapt**: reference only if SB3 friction (project.md §3) |
| 11 | `gradio` 6.x | Apache-2.0 | `gr.Video` scrubs a local mp4 via local server | **adopt**; static HTML `<video>` = mp4-only fallback |
| 12 | `av` (PyAV) 18 | BSD-3 | bundles LGPL-only ffmpeg build; H.264 mp4 | **adopt** for mp4 |
| 13 | `imageio-ffmpeg` 0.6 | BSD-2 wrapper | bundled ffmpeg build flags unstated | **reject** |
| 14 | `opencv-python` VideoWriter mp4v | Apache-2.0 | mp4v poorly browser-compatible | **reject** for video (may still be used for drawing) |
| 15 | `imageio` + Pillow | BSD-2 / MIT-CMU | GIF writing | **adopt** for GIF |
| 16 | ws/100 HPC bundle + sweep launcher + license guard | own code | generic parts named in plan §7 | **adapt** at step 7 / G8 |

## Null results — recorded explicitly
- No Apache/MIT/BSD MuJoCo "camera-drone keeps lock on an evader" env exists. Nearest:
  `Nil69420/mujoco_drones` (quadrotor + sensors, no evasion), `thu-uav/Multi-UAV-pursuit-evasion`
  (different simulator); licences unconfirmed. → building env is justified.
- MuJoCo known issue google-deepmind/mujoco#1121: depth/segmentation ID mismatch when
  overlaying both — we never overlay them (segmentation is not used; GT boxes come from state).

## Numbers the plan will cite
| number | value | source | what it gates |
|---|---|---|---|
| H200 cost vs 4060-derived estimate | ~4× cheaper (P1: 7.74 h actual vs 30–36 h est.) | ws/100 context.md D48+, scope.md §3 | step-6 GPU-h estimate is quoted as a cap |
| P1 total spend | 18.46 / 40 H200-h | ws/100 status.txt | budget discipline reference |
| PERUN site policy | 4 nodes per user at submit time | ws/100 context.md D57b | array `%max_concurrent` |

## Verdict
**Borrow the stack (mujoco, gymnasium, SB3, trackers, gradio, av, imageio), build the env,
sensor, noise injector, policies, harness and demo.** No solved problem is re-solved.
