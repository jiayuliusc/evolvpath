# EvolvPath — shared data contracts

Versioned interfaces between agents. **Changing any of these is a breaking change for consumers**: bump `schema_version`, and update both the producer's and consumer's README. PathFinder (`agents/career-trajectory/README.md`) is the reference implementation and has the full field-level docs.

## 1. Career evidence → Career Trajectory agent (PathFinder)

**Producer:** LinkedIn Data Integration agent (Nic) — or the user directly.

PathFinder accepts four input shapes, all tolerant of missing fields (missing data becomes warnings and lower confidence, never an error):

| Input | How it arrives | Provenance assigned |
| --- | --- | --- |
| Legacy `alumni_history` JSON (`experiences`/`roles`/`positions`, `start_date`/`end_date`, `organisation` aliases OK) | text | `linkedin` (any JSON) |
| Canonical profile JSON (`"schema_version": "2.0"`) | text | `linkedin` |
| Résumé file — PDF, DOCX, TXT or JSON, **inline contents only** (a link is never fetched) | Chat Protocol `ResourceContent` | `resume` |
| Pasted career/résumé text | any message | `user` |

Canonical v2 profile (all fields optional; dates `YYYY-MM`, `YYYY`, `Mon YYYY` or `"present"`):

```json
{
  "schema_version": "2.0",
  "name": "optional", "headline": "optional", "location": "optional",
  "skills": ["SQL", "Python"],
  "education": [{"degree": "BSc", "field": "Bioinformatics", "school": "Northeastern", "end": "2019"}],
  "experiences": [{"title": "Data Analyst", "company": "Acme Labs", "start": "2019-07", "end": "2022-03",
                   "skills": ["SQL"], "achievements": ["Built models lifting forecast accuracy 23%"]}],
  "projects": [{"name": "Forecast toolkit", "description": "...", "skills": ["Python"]}],
  "certifications": [{"name": "CompTIA Data+", "issuer": "CompTIA", "year": "2024"}],
  "target_roles": ["Data Scientist"],
  "preferences": {"timeline_months": 12, "available_hours_per_week": 10,
                  "location": null, "remote_preference": null, "industries": []}
}
```

A sample payload for Nic's integration lives at `agents/career-trajectory/scripts/mock_linkedin.example.json`.

## 2. Career Trajectory → Goal Planning agent (schema 2.0)

**Producer:** PathFinder. **Trigger:** user says "plan my goals" (or taps **Plan my goals**); auto-forwarded when `AGENT_3_ADDRESS` is set on PathFinder. Sent as Chat Protocol `TextContent`; the JSON is also shown copy-ready to the user.

Required fields — **validated before sending; an invalid payload is never sent**:
`schema_version, trajectory_id, profile_id, target_role, scores, selected_path, waypoints, confidence, created_at`

Semantics both sides must honor:

- **Ownership split:** PathFinder owns outcomes, gaps and waypoints; the Goal Planning agent owns all scheduling. Schedule-like fields (`daily`/`weekly`/`schedule`) are rejected by validation.
- **Privacy:** no name/headline/location leaves PathFinder — only the anonymous `profile_id` (`anon-xxxxxxxx`).
- `scores`: `readiness`, `transferability`, `evidence_strength` = integers 0–100; `transition_difficulty` = `low|medium|high`; `confidence` = 0–1 (evidence quality, **not** probability of any outcome).
- `skill_gaps[]`: each has `skill`, `current_level`, `required_level`, `gap_type`, `priority`, `source`. Gap types:
  - `confirmed_gap` — real evidence exists but is below the requirement
  - `missing_evidence` — nothing on file either way (**not** inability)
  - `unverified` — only list-level/inferred support
- `waypoints[]`: `{"outcome", "target_horizon", "success_criteria"}` — outcome-first horizons like `"0-3 months"`, never dated schedules.
- Proficiency scale: 0 none, 1 awareness, 2 working, 3 proficient, 4 expert.
- Every number is deterministic scoring output (no LLM in the math) and reproducible for identical inputs + `as_of` date.

## 3. Agent identity & addressing

- Addresses are derived from each agent's `AGENT_SEED`: **one seed per agent, kept forever** (a new seed = a new identity).
- Cross-agent messages are Chat Protocol `ChatMessage`s sent with `ctx.send(address, …)`; the target address comes from that agent's env (e.g. `AGENT_3_ADDRESS`).
- A mailbox agent is only reachable while its process runs locally — plan the joint demo accordingly.
