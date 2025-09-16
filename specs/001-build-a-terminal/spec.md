# Feature Specification: Column Popper Game and RL Environment

**Feature Branch**: `001-build-a-terminal`
**Created**: 2025-09-15
**Status**: Draft
**Input**: User description: "Build a terminal-playable Column Popper puzzle that also functions as a Gym-compatible RL environment. The game spawns numbers that fall each tick; the player or agent picks up from a column and drops into another to make vertical triples that pop for points, with overflow or time-up ending the run. It supports human mode via curses UI and headless mode -fast simulation, exposes clear observations/actions/rewards, and is deterministic when seeded. Why: to practice end-to-end RL environment design, get immediate human feedback on difficulty and rewards, and provide a lightweight, reproducible sandbox where simple heuristics and standard RL baselines can be benchmarked quickly."

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation
When creating this spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - User types and permissions
   - Data retention/deletion policies
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a player, I want to play a column-based puzzle game in my terminal where I can move numbers to create vertical matches and score points before the board fills up or time runs out. As an RL researcher, I want to use this same game as a Gym-compatible environment to train and benchmark reinforcement learning agents.

### Acceptance Scenarios
1. **Given** a new game is started in human mode, **When** the player picks up a number from one column and drops it into another, **Then** the number is moved, and if a vertical triple is formed, the triple is removed and the score increases.
2. **Given** the game is running in headless mode as a Gym environment, **When** an agent takes an action to move a number, **Then** the environment state is updated, a reward is issued, and the next observation is provided.
3. **Given** the game board is almost full, **When** a number falls and there is no space in its column, **Then** the game ends due to overflow.
4. **Given** the game has a time limit, **When** the timer reaches zero, **Then** the game ends.
5. **Given** a seed is provided to the environment, **When** the same sequence of actions is taken, **Then** the game progression and outcomes are identical across multiple runs.

### Edge Cases
- What happens when a player tries to pick from an empty column?
- What happens when a player tries to drop into a full column?
- How does the system handle a game end and restart sequence in both human and headless modes?

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: The system MUST provide a terminal-based game interface using curses, with an ASCII fallback when curses is unavailable.
- **FR-002**: The game MUST spawn numbers that fall down the columns on a schedule; the player/agent can also trigger a manual fall.
- **FR-003**: Players/agents MUST be able to pick up the topmost number from any column and later release it onto a target column with space.
- **FR-004**: Players/agents MUST be able to drop a held number into any column with available space.
- **FR-005**: The system MUST detect and remove any vertical sequence of three identical numbers in the acted column (“pop”).
- **FR-006**: Scoring MUST follow the constitutional defaults: +1 per valid player action; +3 per popped cell (i.e., +9 for a triple); −0.01 step cost; −1 on overflow or invalid full-column drop; −0.5 on time-up truncation.
- **FR-007**: The game MUST end if any column overflows (a new number spawns with no space) or, when `strict_invalid=True`, on an invalid drop.
- **FR-008**: The game MUST end when a time limit is reached; default `GAME_DURATION` is 60 seconds and is configurable. Fall rate follows a schedule.
- **FR-009**: The system MUST support a headless mode for fast, non-visual simulation.
- **FR-010**: The system MUST expose a Gym-compatible API, including observation space, action space, and reward signals.
- **FR-011**: The game state MUST be deterministic when initialized with a specific seed.
- **FR-012**: The observation for the RL agent MUST be a Dict with at least: `board` (int32 grid 12×3, 0=empty, values from number_pool), and `selection` (shape (2,) → [is_selected, value]); optionally `time_left_norm` in [0,1].
- **FR-013**: The action space MUST be `Discrete(4)` with: 0–2 → select/release on target column; 3 → manual fall.
- **FR-014**: The reward function MUST follow FR-006 by default; alternate reward presets may be offered behind an explicit configuration flag.
- **FR-015**: The board MUST be 12 rows by 3 columns; `number_pool` defaults to [1,2,3]; the spawn generator MUST avoid creating an immediate three-in-a-column from two identical cells directly beneath (per-column constraint).

### Key Entities *(include if feature involves data)*
- **Game Board**: A 2D grid representing the play area, containing numbers or empty spaces.
- **Player/Agent**: The actor interacting with the game, either human or AI.
- **Number**: The game pieces that fall and are moved. They have a value (the number itself).
- **Score**: A numerical value representing the player's points.
- **Timer**: A countdown mechanism to limit the game duration.

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified

---
