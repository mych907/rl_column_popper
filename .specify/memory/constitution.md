# Spec Kit AI Constitution

## Core Principles

### I. Spec‑First, Prototype‑Next (Non‑Negotiable)

Every feature begins as a concise, testable spec. We write the contract (types, I/O, invariants, failure modes) before code. Prototypes exist only to validate a spec and must be discarded or elevated into production via tests.

### II. CLI + Text/JSON I/O Everywhere

All libraries and environments expose a CLI with pure text/JSON streams: stdin/args → actions/events; stdout → observations/rewards; stderr → logs. This enables scripting, reproducibility, and language‑agnostic integration.

### III. Test‑First, Deterministic by Default

TDD rules the loop: write failing tests, make them pass, refactor. All randomness is seeded and logged; any nondeterminism must be explicitly opted‑in and measurable.

### IV. Gym‑Compatibility as a Contract

Environments implement Gymnasium’s API faithfully (reset, step, render, close), declare `action_space` and `observation_space`, and provide wrappers for vectorized rollout, logging, and curriculum.

### V. Observability and Traceability

Structured event logs (JSON Lines), episode traces, and reproducible run manifests (seed, version, hashes). Time‑to‑step, GC, and FPS metrics are first‑class for profiling RL loops.

### VI. Semantic Versioning + Safe Evolution

Use MAJOR.MINOR.PATCH. Breaking changes require migration notes, deprecated shims, and test updates. Observation schemas carry version tags.

### VII. Simplicity First, Extensibility Second

Keep surface area small (YAGNI). Provide narrow interfaces and plug‑points (reward, scheduler, generator) rather than deep inheritance hierarchies.

### VIII. Human‑Playable by Design

Every env must support a zero‑install terminal UI (curses/ANSI) and an accessibility fallback (pure ASCII, no special keys). Human playtests are part of CI.

### IX. Cross‑Platform Portability

Linux/macOS/Windows support with no GPU requirement. Optional acceleration must degrade gracefully. Zero hidden system dependencies.

### X. Fairness and Safety for RL Research

Disallow reward leakage and hidden state. Provide standard evaluation seeds and locked configs for fair comparisons. Document failure modes and edge cases.

## Product Spec: Column Popper (Terminal + Gym Env)

### What it is

A fast, deterministic, terminal‑playable puzzle where numbers fall in 3 columns on an 8×3 board. The player can pick up a number from a column and drop it into any column; three identical numbers vertically adjacent “pop.” The same binary is a Gymnasium environment for RL.

### Why it exists

1. A minimal but nontrivial RL benchmark: delayed consequences, stochastic generation with local constraints, and time pressure. 2) A teaching scaffold for writing Gym envs from scratch with strong testing and CLI contracts. 3) A clean demo for spec‑driven development.

### Rules and Contracts (authoritative)

* Board: ROWS=8, COLS=3. Integers 0 (empty) or from a `number_pool` (default \[1,2,3]).
* Actions (Discrete 4): 0,1,2 → select/release on target column; 3 → manual fall (advance time one row).
* Invalid actions: treated as no‑op with a small penalty unless `strict_invalid=True` (then episode terminates with a fault code).
* Observation (Dict):

  * `board: Box(low=0, high=9, shape=(8,3), dtype=int32)`
  * `selection: Box(low=0, high=9, shape=(2,), dtype=int32)` → \[is\_selected, value]
  * Optional feature flags (gated): `time_left_norm` in \[0,1].
* Info keys: `score`, `time_left`, `pops_this_step`, `fall_interval`, `seed`, `version`.
* Reward (default): +1 per valid player action, +3 per popped cell (i.e., clearing a triple yields +3), −0.01 step cost, −1 on overflow or invalid drop, −0.5 on time‑up truncation. Alternate reward presets are pluggable.
* Episode end: `game_over` (piece falls off or invalid full‑column drop) or `timer<=0` (truncation). Default `GAME_DURATION=60s` with dynamic fall rate schedule.
* Generation constraint: the new top row avoids creating an immediate three‑in‑a‑column from prior two identical cells directly beneath. This is enforced per column.
* Render modes: `human` (curses), `ansi` (string frame for unit tests and headless logs), optional `rgb_array` (for notebooks).
* Performance budget: `<100 µs` median `env.step()` on 3.5 GHz desktop for default sizes; `<2 ms` frame render in `ansi`.

### CLI Contract (reference)

```
column_popper --mode=play --seed=42             # human terminal
column_popper --mode=rollout --episodes=100 \
  --format=jsonl --seed=42 --reward=preset:v1   # headless evaluation
# stdin: lines of actions (0..3); stdout: JSONL of observations and rewards
```

### Invariants (test as property)

* Pops only remove triples and only in the acted column.
* No negative numbers or values outside `number_pool` appear.
* Selecting and replacing preserves multiset of numbers unless a pop occurs.
* With fixed seed and action sequence, observation sequence is identical (determinism).

## Engineering Constraints & Quality Bar

* 100% type coverage for public API (mypy strict). Unit coverage ≥ 90% core; property tests via Hypothesis for generators/invariants.
* Lint: Ruff + Black; pre‑commit enforced in CI.
* Reproducibility: `RunManifest` persisted (seed, git SHA, config hash, env version, platform).
* Benchmark CI gate: fail if `step()` p95 exceeds budget by >20% on reference machine.
* Accessibility: `--no-curses` ASCII fallback; avoids terminals without keypad support.

## Development Workflow

* Design: each feature starts with a one‑page spec (problem, contract, tests, risks). For disagreements, write an ADR (architecture decision record).
* Reviews: two approvals required; checklist includes determinism, logging, docs, and migration notes.
* CI stages: lint → type → unit → property → integration (CLI protocol) → human‑play smoke (scripted input) → benchmarks.
* Release: SemVer tags, changelog, upgrade notes, schema version bump if observation changes.

## Roadmap (initial)

* M0: Spec and skeletal env with `ansi` renderer and deterministic generator.
* M1: Curses UI and CLI harness; JSONL rollout tool.
* M2: Curriculum and reward presets; vectorized env wrapper.
* M3: Evaluation suite with fixed seeds and leaderboards; example PPO script.

## Non‑Goals (for v1)

* No networked multiplayer.
* No external dependencies for graphics beyond curses.
* No hidden auto‑play bots in the production binary.

## Risks & Mitigations

* Reward hacking via unintended loops → property tests on invariants; adversarial fuzzing.
* Terminal portability issues → ASCII fallback and CI on Windows/macOS/Linux.
* Curriculum overfitting → publish fixed evaluation configs and seeds.

## Governance

This constitution supersedes ad‑hoc practices. Amendments require: 1) updated spec, 2) migration notes, 3) passing CI including reproducibility checks. All PRs must link to either a spec or ADR. Observation schema changes require a minor or major version bump and deprecation shims when viable.

**Version**: 1.0.0 | **Ratified**: 2025‑09‑15 | **Last Amended**: 2025‑09‑15