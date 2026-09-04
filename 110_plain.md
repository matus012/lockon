# 110_lockon — plain-English overview (v1)
Self-contained. Plain language, no code. Supersedes v0.

## What it is
A simulated drone-camera that learns to MOVE so it never loses sight of a person who is actively
trying to hide — behind walls, in the dark, when a sensor dies. You watch it play out as short clips
plus a chart showing exactly where it breaks.

## Why
The last project (P1) failed on the one piece we didn't control: spotting people in real footage.
This one is built so nothing outside our control can stop it. Everything happens inside a simulator
we own. Goal = a demo strong enough to be the centerpiece for interviews. Not solving a real-world
problem, not transferring to real robots.

## What gets built (7 pieces, each works alone)
- **Arena** — a 3D room with pillars and walls, lights you can switch off, a drone-camera that
  moves, a person that moves. Built in MuJoCo (free, open, standard physics simulator).
- **Senses** — three views of the same scene: color, depth, and a fake "heat" view. Any one can be
  switched off to simulate a broken sensor.
- **Eyes** — the tracker. Follows the person and keeps the same ID even when they vanish briefly.
  Uses the simulator's own perfect knowledge of where the person is, with realistic noise added,
  so it can never "fail to see" for reasons we can't fix. Standard, proven tracker (ByteTrack).
- **Prey (scripted)** — the person. Hand-written rules: run for cover, seek darkness, break line of
  sight. Difficulty dial from easy to nasty. This is the examiner, not a learner.
- **Hunter (learned)** — the drone. Learns by trial and error (reinforcement learning, standard
  PPO) to position itself so the tracker keeps the lock. Rewarded for seeing, penalized for losing.
  A simple rule-based hunter exists too — the guaranteed fallback if learning disappoints.
- **Examiner** — runs hundreds of games while dialing up darkness, walls, speed, sensor loss, and
  draws the curves: fixed camera vs rule-based hunter vs learned hunter.
- **Stage** — the viewer. Three showcase clips plus the curves, pre-rendered so they always play.

## The three showcase moments
1. Person ducks behind a pillar → drone repositions → lock survives.
2. Lights cut → color view goes black → heat view takes over → lock survives.
3. The chase: smart hunter vs hard prey, full run.

## Where things come from
- Simulator, tracker, learning library, viewer: all free and open (Apache/MIT).
- Training data: none needed. The simulator generates its own games; the hunter learns from playing.
- No real footage, no external datasets, no licensing strings. Public release is fine.

## Where it runs
- **Laptop (days 0–4):** the dev box — building, testing, first learning runs (max 5 h each,
  resumable). The finished demo runs on the laptop forever.
- **Cluster (from day 2, budget 70–100 GPU-hours + CPU hours):** joins once the hunter's learning
  code passes its laptop test. Runs many learning attempts in parallel on CPU nodes, picks the best,
  adds error bars to the curves; GPU hours only for heavy rendering and a short learned-prey trial.
  Big files (videos, checkpoints, 1 TB) stay on the cluster; only code, GIFs, curves and one small
  best-model file come home (30 Mbps home line). Result: "trained on H200 cluster" line in README.
  An extra 300 hours is a post-ship option only if it would fundamentally improve the demo.

## Timeline and what you do
- **Evening 0 (~1 h):** approve plan, paste handoff to Claude Code, one-time laptop setup, leave.
- **Day 1 (5 min):** read status line, glance at first clips, reply "continue".
- **Day 2 (10 min):** learning code passed → approve cluster launch (one line). Cluster sweeps
  start while laptop keeps building.
- **Day 3 (5 min):** status: jobs alive, best score, hours spent.
- **Day 4 (30 min):** review three clips + curves with error bars. Decide: ship / fix / +300 h.
- **Ship (15 min):** approve making it public. Optional one-hour promotion push.
Total your time: ~3 h over 4–5 days. Everything else runs unattended.

## Safety rails
- Each piece is standalone and reusable; nothing is welded together.
- Every stage has a working fallback, so a demo always ships.
- Learning never depends on rendering — it learns from positions, not pixels — so it's fast and
  can't be slowed by graphics.
- You approve only: making the repo public, spending money, deleting anything.
- Claude Code manages its own notes (context, status, per-piece logs). You read one status line.
- It runs until a gate passes or a fallback fires — both are automatic. It stops for you ONLY on:
  approval-list items (public / money / delete), a fallback that also failed, or a decision that
  changes what the demo is. Everything else it resolves itself and logs.