# Tasks: Column Popper Game and RL Environment

**Input**: Design documents from `/mnt/c/Git/column_popper_spec_driven/specs/001-build-a-terminal/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root

## Phase 3.1: Setup
- [x] T001 Create project structure in `src/column_popper` and `tests/` per implementation plan
- [x] T002 Initialize Python project with `pyproject.toml` and install dependencies: `gymnasium`, `numpy`, `windows-curses`, `argparse`, `jsonlines`, `pytest`, `hypothesis`, `pytest-benchmark`, `mypy`, `ruff`, `black`, `hatchling`
- [x] T003 [P] Configure `ruff` and `black` in `pyproject.toml`
- [x] T004 [P] Configure `mypy` in `pyproject.toml`
- [x] T005 [P] Create `src/column_popper/version.py` with `__version__ = "0.1.0"`

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T006 [P] Contract test for Gymnasium environment in `tests/contract/test_gym_env.py`. It should check the observation space, action space, and reward range.
- [x] T007 [P] Integration test for a full game loop in `tests/integration/test_game_loop.py`. It should cover picking up, dropping, popping, and game over conditions.
- [x] T008 [P] Unit test for the board logic in `tests/unit/test_board.py`. It should test adding numbers, checking for pops, and removing numbers.

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [x] T009 [P] Implement the `Board` class in `src/column_popper/core/board.py`.
- [x] T010 [P] Implement the `Schedule` class in `src/column_popper/core/schedule.py` to manage game ticks.
- [x] T011 Implement the Gymnasium environment in `src/column_popper/envs/column_popper_env.py`.
- [x] T012 Implement the ANSI renderer in `src/column_popper/render/ansi.py`.
- [x] T013 Implement the curses UI in `src/column_popper/render/curses_ui.py`.
- [x] T014 Implement reward presets in `src/column_popper/rewards/presets.py`.
- [x] T015 Implement the CLI for playing the game in `src/column_popper/cli/play.py`.
- [x] T016 Implement the CLI for rolling out a policy in `src/column_popper/cli/rollout.py`.
- [x] T017 Implement the CLI for streaming data in `src/column_popper/cli/protocol.py`.
- [ ] T018 Implement utilities for run manifests in `src/column_popper/utils/manifest.py`.
- [ ] T019 Implement utilities for random number generation in `src/column_popper/utils/rng.py`.

## Phase 3.4: Integration
- [ ] T020 Integrate the `Board` and `Schedule` into the `ColumnPopperEnv`.
- [ ] T021 Integrate the ANSI and curses renderers with the game loop.
- [ ] T022 Integrate the reward presets with the environment.
- [ ] T023 Integrate the CLI commands with the game environment.

## Phase 3.5: Polish
- [ ] T024 [P] Add unit tests for all new classes and functions.
- [ ] T025 [P] Add performance benchmarks for the `step` function.
- [ ] T026 [P] Add documentation and type hints to all public APIs.
- [ ] T027 Run `ruff check .` and `mypy .` to ensure code quality.

## Dependencies
- Setup (T001-T005) before everything.
- Tests (T006-T008) before implementation (T009-T019).
- Core implementation (T009-T019) before integration (T020-T023).
- Integration (T020-T023) before polish (T024-T027).

## Parallel Example
```
# Launch T003, T004, and T005 together:
Task: "Configure ruff and black in pyproject.toml"
Task: "Configure mypy in pyproject.toml"
Task: "Create src/column_popper/version.py with __version__ = "0.1.0""

# Launch T006, T007, and T008 together:
Task: "Contract test for Gymnasium environment in tests/contract/test_gym_env.py"
Task: "Integration test for a full game loop in tests/integration/test_game_loop.py"
Task: "Unit test for the board logic in tests/unit/test_board.py"
```

```
