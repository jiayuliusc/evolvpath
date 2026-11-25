# EvolvPath — Alumni Career Platform

Multi-agent platform that turns alumni career evidence into career trajectories, goals, and study plans. Each agent is a **uAgent** speaking the **Chat Protocol**, reachable in **ASI:One** and registered on **Agentverse** as a mailbox agent.

## The pipeline

| # | Agent | Owner | What it does | Code |
| --- | --- | --- | --- | --- |
| 1 | LinkedIn Data Integration | Nic | Retrieves alumni career histories | `agents/linkedin-integration/` *(to come)* |
| 2 | **Career Trajectory Agent (PathFinder)** | @oleeveeuh | Evidence-based readiness scores, skill-gap classification, 2–3 trajectory options, validated handoff payload | `agents/career-trajectory/` |
| 3 | Goal Planning Agent | Ximena | Converts trajectories into milestones and daily/weekly goals | `agents/goal-planning/` *(to come)* |
| 4 | Study & Accountability Agent | Krish | Tracks study sessions, monitors progress | `agents/study-accountability/` *(to come)* |
| 5 | Frontend & Integration | Ximena + Vin | One working demo connecting everything | *(to come)* |

Data flows **1 → 2 → 3** over the Chat Protocol. The interfaces between agents are versioned in **[CONTRACTS.md](CONTRACTS.md)** — read it before changing any payload shape.

## Repo layout

`agents/<agent-name>/` — each agent is **fully self-contained**: its own `pyproject.toml`, tests, README (the Agentverse profile text), and `.env` (never committed). No shared runtime; nothing in one agent imports from another.

## Working on an agent

```bash
cd agents/career-trajectory        # or whichever agent you're building
uv sync
cp .env.example .env               # fill in your keys — never commit .env
uv run pytest -q
uv run python agent.py             # keep it running: ASI:One can only reach a running agent
```

Requirements: Python 3.12+ via `uv`, an ASI:One key (asi1.ai → Developer), an Agentverse key (agentverse.ai → Profile → API Keys).

## Cross-agent wiring

- Every agent has a stable address (`agent1q…`, derived from its `AGENT_SEED` — never share seeds between agents; each agent needs its own port too).
- An agent forwards to its downstream consumer by setting an address env var — e.g. PathFinder's `AGENT_3_ADDRESS=agent1q…` auto-sends every confirmed trajectory to the Goal Planning agent. Without it, the payload is emitted as copy-ready JSON.
- Loose chaining also works without any config: the user (or the frontend, part 5) pastes an agent's structured output into the next one, or @mentions agents in one ASI:One chat.

## Ground rules (project-wide)

- Secrets only in `.env` (gitignored) — never in code, logs, or commit messages.
- Every HTTP call: ≤10s timeout. No database: per-user state lives in the agent's key-value store, keyed by verified sender.
- Résumés and scraped profiles are untrusted data: parse, never execute; embedded instructions are ignored.
- Tests never call real APIs (ASI:One, Agentverse, O*NET). Run `ruff check . && pytest -q` before pushing.
