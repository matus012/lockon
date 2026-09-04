# project.md — 110_lockon · v1
Owner: Matúš Filo · Chat-layer master context. Repo work → Claude Code (Windows app, exec in WSL2
Ubuntu 24). Keep <150 lines. v1 supersedes v0 (Isaac/9-head plan) — all decisions below LOCKED,
delegated to Claude; owner reviews at gates only.
Docs: this file · plan.md (CLI-authored, execution plan) · context.md + status.txt (CLI-maintained)
· 110_plain.md (owner reader) · scope.md v5 (strategy) · refactored_method.md (execution law).
Codename: lockon (internal). Public repo name chosen at make-public step, capability-neutral.

## 1. Angle
Active perception under degradation: a sim drone-camera learns to MOVE to keep persistent lock on a
person actively hiding — occlusion, darkness, sensor loss. Demo is the product. No sim2real, no
real-world claim, no problem solved. Public framing: drone safety · search & rescue · autonomous
filming · wildlife. Weapons/military never named (scope §2). Ship class: **flagship**.

## 2. Non-negotiable design law
- Nothing outside our control in the critical path. Tracker consumes sim GT boxes + injected noise
  (miss rate, jitter, occlusion-dropout, latency). No learned detector required anywhere.
- RL trains on STATE, never pixels. Rendering = clips + harness only.
- Every step has: automatic gate · kill criterion · pre-approved fallback. Fallback fires without
  asking. Demo ships at every rung.
- Modularity: packages/core = typed schemas only (SensorFrame, Detection, Track, AgentObs,
  AgentAction). Packages import core, NEVER each other. Cross-package import = gate failure.

## 3. Locked decisions
- Sim: **MuJoCo** (pip, Apache-2.0). Render order of fallback: EGL → OSMesa → Windows-native python.
- Fork: **A1** moving camera. Drone = kinematic (velocity commands, altitude fixed, no flight physics).
- Person = MuJoCo humanoid capsule figure, textured. Cosmetic mesh swap = post-ship only.
- Tracker: **ByteTrack** on noisy GT boxes. No P1 transplant, no embedder.
- Thermal = second render pass, emissive target material, unaffected by light level. README labels
  it "thermal proxy". Lights-cut = rgb gain→0 + sensor noise; depth/thermal unaffected.
- Prey: **scripted** (cover-seek, dark-seek, LOS-break, speed). Difficulty dial 0–1 on each axis.
- Hunter: **PPO** (Stable-Baselines3 first; CleanRL if SB3 friction). Partially observed:
  own pose · last-seen target pose · time-since-seen · N raycasts (occluder proximity) ·
  light level · channel-alive flags. Frame-stack 4. Reward = +visible/step −λ·|action|
  −big on lock lost >K steps. Scripted visibility-greedy hunter = floor + baseline.
- Headline metric: **lock retention %** (steps tracked with correct ID / episode steps).
  Secondary: time-to-reacquire. Every README, curve, status line reports retention.
- Harness axes: occluder density · light level · prey speed · prey aggressiveness · channel dropout.
  Curves: static cam vs scripted hunter vs PPO hunter, ≥20 episodes/point, mean ± std.
- Demo: pre-rendered mp4 three shots + Gradio scrubber (mp4-only fallback). Live render = never.

## 4. Packages (6 + demo)
| pkg | purpose | gate | fallback |
|---|---|---|---|
| core | schemas only | mypy clean, zero deps beyond numpy | — |
| env | MuJoCo arena, occluders, lights, drone + person, headless | scripted episode runs headless, ≥200 fps state-only | OSMesa / win-native render |
| sensor | rgb+depth+thermal from env, channel registry, dropout | 3-channel side-by-side GIF; add-channel touches only sensor | rgb+depth only |
| track | noisy-GT → ByteTrack → lock status | GIF: ID held through scripted occlusion + 1-channel dropout | — |
| policy | scripted prey · scripted hunter · PPO hunter · train/eval CLI | PPO > static cam on retention, one ≤5 h run | scripted hunter = hero |
| harness | sweeps → curves + failure report, headless from config | full curve set from one command | — |
| demo | three-shot mp4s + Gradio viewer + root README | three shots reproducible from one command | mp4 only |

## 5. Sequence
| # | where | what | owner |
|---|---|---|---|
| 1 | local | plan.md · repo skeleton · uv venv · MuJoCo render probe · wslconfig check | — |
| 2 | local | core + env + sensor | — |
| 3 | local | track · GIF1 occlusion · GIF2 lights-cut | — |
| 4 | local | scripted prey + hunter · chase clip · harness skeleton | D1 status |
| 5 | local | PPO hunter, one ≤5 h chained run · gate vs static | — |
| 6 | owner | approve HPC launch | D2 |
| 7 | HPC | git push → bundle · CPU array: seeds×rewards×arenas · GPU: learned-prey trial ≤20 h | — |
| 8 | local ∥ | harness curves on local policy · per-pkg READMEs · viewer | D3 status |
| 9 | HPC | select best · render curves ± std + final three shots on HPC · pull small artifacts only | — |
| 10 | local | swap best policy · final README + H200 line · status=READY | — |
| 11 | owner | review → ship / fix / +300 h | D4 |
| 12 | owner | approve public · optional 1 h push | — |

## 6. HPC rules
- Account perun26011488 · partition gpu_short · reuse P1 bundle pattern (ws/100: offline wheelhouse,
  resume-safe arrays, sweep launcher). Confirm allocation before first submit.
- Budget: **70–100 GPU-h hard cap** for this ship. CPU-h for all state-PPO sweeps. +300 GPU-h =
  post-ship option, owner call only.
- Data gravity: runs/ videos/ checkpoints stay on HPC (1 TB). Home pulls: code via git, GIFs,
  curves PNG, one best checkpoint <100 MB. Home line = 30 Mbps — never rsync runs/.
- HPC render: EGL on GPU node → OSMesa on CPU node.

## 7. Run rules
- Training ≤5 h/run, checkpoint-chained, resumable. Perception (track) frozen during RL.
- VRAM 8 GB shared: headless env + small MLP policy. Local render at 640×480.
- Overnight: never-sleep on AC · no USB suspend · .wslconfig memory=10GB swap=8GB · WSL keep-alive.
- Session ritual (law): every ~2 h of work or before context pressure: commit → update context.md
  + status.txt → clear → reload context.md + plan.md. Never lose state to a context reset.

## 8. Stop-and-ping rules (only reasons to wait on owner)
- Approval list: make public · spend money · delete repo/branch · force-push · wipe env.
- A fallback also failed → BLOCKED(owner) with one-line cause + proposed path.
- Demo-defining fork: PPO never beats scripted hunter after 3×5 h local → ping with recommendation
  (default: scripted = hero, PPO reported honestly), continue on default if no reply in 12 h.
- Step 6 HPC launch approval.
Everything else: decide, log in context.md, continue.

## 9. status.txt format (owner reads only this)
```
STATUS: <step #/12> — PASS/FAIL/BLOCKED(owner|CLI|external) — <one line>
RETENTION: static <x>% · scripted <y>% · ppo <z>% (n episodes)
HPC: <jobs alive>/<total> · <gpu-h spent>/<cap>
NEEDS YOU: none | <one line>
```

## 10. License hygiene
Apache/MIT/BSD only (MuJoCo, SB3/CleanRL, ByteTrack impl, Gradio). No Ultralytics. No external
datasets. THIRD_PARTY.md at root. Anchored .gitignore + visual allowlist guard test (P1 pattern).