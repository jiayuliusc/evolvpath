![tag:chatprotocol](https://img.shields.io/badge/chatprotocol-1D3BD4)

# PathFinder — Career Trajectory Agent

PathFinder turns career evidence — a LinkedIn-derived history, a résumé (PDF, DOCX, TXT, JSON), or pasted experience — into an evidence-based career trajectory: readiness scores against a target role, classified skill gaps, two or three compared path options, and a recommended trajectory the Goal Planning agent can turn into milestones and goals.

Scores are career-planning guidance, not hiring eligibility or candidate ranking, and no employment, salary, promotion or completion outcome is ever guaranteed.

## What it can do
- Parse `alumni_history` JSON (legacy format, still fully supported), canonical v2 profile JSON, résumé files, or pasted free text into a canonical profile with provenance
- Reconcile LinkedIn history with résumé evidence: deduplicate positions, keep provenance (`resume` / `linkedin` / `user` / `onet`), and flag conflicting titles or dates for your confirmation instead of silently dropping them
- Classify every skill as `demonstrated`, `applied`, `mentioned` or `inferred` — demonstrated evidence counts most, and a job title never invents a skill
- Benchmark a target role against O*NET (skills with required levels, tasks, job zone, education expectations, outlook), recording source and retrieval date
- Score readiness deterministically (no LLM in the math): readiness, transferability, evidence strength, transition difficulty, per-category breakdown, assumptions, confidence
- Distinguish a **confirmed capability gap** from **missing evidence** from an **unverified** claim — missing résumé evidence is not treated as missing ability
- Generate and compare 2–3 trajectory options (direct / skill-bridge / intermediate-role / longer reskilling) with waypoints, risks and constraint fit
- Run what-if scenarios (certification, degree, relocation, remote-only, accelerated schedule, different target roles) reusing the same engine
- Hand a validated, versioned payload to the Goal Planning agent — copy-ready on request, auto-forwarded when `AGENT_3_ADDRESS` is set

## Example prompts
- `{"name": "Maya Chen", "experiences": [{"title": "Data Analyst", "company": "Acme Labs", "start": "2019-07", "end": "2022-03"}], "target_role": "Data Scientist"}` — what trajectory should I follow?
- I have 4 years as a data analyst — analyze my readiness for data scientist
- Here is my résumé (attach a PDF/DOCX or paste the text) — which path should I take?
- What if I add a certification — or targeted data engineering instead?

## Supported inputs
| Input | How | Provenance |
| --- | --- | --- |
| `alumni_history` JSON (v1 aliases: `roles`/`positions`, `start_date`/`end_date`, `organisation`) | paste or from the LinkedIn integration | `linkedin` |
| Canonical v2 profile JSON (`schema_version: "2.0"`) | paste | `linkedin` |
| Résumé file | attach PDF, DOCX, TXT or JSON (inline contents only — remote URLs are never fetched) | `resume` |
| Pasted career text / résumé text | any message | `user` |

All fields are optional. Missing data produces warnings and lower confidence — never crashes or fabricated facts.

## Résumé ingestion
Extraction (PDF via pypdf, DOCX via python-docx, text/JSON natively), hygiene (control-character stripping, size caps, instruction-like text detected and ignored as data), section detection, career-fact extraction, normalization and validation are separate stages. Unsupported or undecodable formats return a clear structured response rather than a pretend parse. Résumé content is treated strictly as untrusted career data: nothing is executed and embedded instructions are never followed. Raw file text never reaches the LLM or the logs — only extracted facts do.

## Canonical input schema (v2.0)
```json
{
  "schema_version": "2.0",
  "profile_id": "anonymous-profile-id",
  "name": "optional",
  "headline": "optional",
  "location": "optional",
  "skills": ["SQL", "Python"],
  "education": [{"degree": "MSc", "field": "Statistics", "school": "Northeastern", "end": "2019"}],
  "experiences": [{"title": "Senior Data Analyst", "company": "Acme Labs", "start": "2022-03",
                   "end": "present", "skills": ["Python"],
                   "achievements": ["Built regression models lifting forecast accuracy 23%"]}],
  "projects": [{"name": "Forecast toolkit", "description": "...", "skills": ["Python"]}],
  "certifications": [{"name": "CompTIA Data+", "issuer": "CompTIA", "year": "2024"}],
  "target_roles": ["Data Scientist"],
  "preferences": {"industries": [], "location": null, "remote_preference": null,
                  "salary_target": null, "timeline_months": 12, "available_hours_per_week": 10},
  "constraints": [],
  "source_metadata": []
}
```
Dates are `YYYY-MM`, `YYYY`, `Mon YYYY` or `"present"`. Fields map to a canonical profile with anonymous `profile_id` (`anon-xxxxxxxx`).

## Scoring methodology (deterministic, documented, configurable in `scoring.py`)
| Category | Weight | Missing-data handling (documented assumptions) |
| --- | --- | --- |
| Relevant experience | 30% | undated roles contribute no years; noted in assumptions |
| Skill coverage | 25% | evidence-weighted match against O*NET skills; neutral 60 when the benchmark lacks skills |
| Achievement evidence | 15% | quantified outcomes + demonstrated benchmark skills + projects |
| Seniority and scope | 10% | estimated seniority band vs O*NET job zone; neutral when zone unavailable |
| Domain alignment | 10% | title/project vocabulary overlap; neutral on sparse text |
| Education and certifications | 5% | only when the benchmark states a requirement; no education on file scores 50, not 0 |
| Recency | 5% | decays ~1.5 pts/month after a 6-month grace window; neutral 50 when no dates exist |

Outputs: `readiness_score`, `transferability_score`, `evidence_strength_score`, `transition_difficulty` (low/medium/high), per-category breakdown with evidence/gaps/assumptions, and a 0–1 `confidence`. Scores are integers 0–100, reproducible for identical normalized inputs, rounded — no false precision. Thresholds: difficulty is `low` at readiness ≥75 with no confirmed high gaps, `medium` at ≥55 (≤2 high gaps), else `high`.

## Evidence levels
| Level | Meaning | Weight | Proficiency level (0–4 scale) |
| --- | --- | --- | --- |
| `demonstrated` | project, quantified outcome, concrete result | 1.0 | 3 (4 when quantified) |
| `applied` | explicitly used in a role | 0.7 | 2 |
| `mentioned` | skills list / summary only | 0.4 | 1 |
| `inferred` | context-suggested, not confirmed | 0.15 | 1 |

The 0–4 proficiency scale: 0 none, 1 awareness, 2 working, 3 proficient, 4 expert. Required levels come from O*NET importance (≥85→4, ≥70→3, ≥50→2, ≥30→1).

## Gap types
| `gap_type` | Meaning |
| --- | --- |
| `confirmed_gap` | real evidence exists but sits below the requirement, or the profile is detailed (outcome evidence + completeness ≥ 0.5) so absence is telling |
| `missing_evidence` | nothing on file either way — **not** treated as inability |
| `unverified` | only list-level or inferred support |

Each gap carries `skill`, `current_level`, `required_level`, `gap_type`, `priority` (high/medium/low), `evidence` and `source`.

## Goal Planning output (schema 2.0)
Say "plan my goals" (or tap **Plan my goals**) after selecting a path. The payload contains **no name, headline or location** — only the anonymous `profile_id` — and no schedules: PathFinder owns outcomes, gaps and waypoints; the Goal Planning agent owns scheduling.

```json
{
  "schema_version": "2.0",
  "trajectory_id": "traj_1a2b3c4d5e",
  "profile_id": "anon-9f8e7d6c",
  "trace_id": "trace_1a2b3c4d",
  "target_role": "Data Scientist",
  "scores": {"readiness": 66, "transferability": 71, "evidence_strength": 55, "transition_difficulty": "medium"},
  "selected_path": {"title": "Skill-bridge into Data Scientist", "roles": ["Senior Data Analyst", "Data Scientist"],
                    "estimated_months": 9, "rationale": "...", "confidence": 0.75},
  "alternatives": [{"id": "path-2", "title": "Direct transition", "type": "direct_transition"}],
  "strengths": ["Python (level 3/4, demonstrated)"],
  "skill_gaps": [{"skill": "Machine learning", "current_level": 0, "required_level": 2,
                  "gap_type": "confirmed_gap", "priority": "high", "evidence": [], "source": "O*NET Web Services v2"}],
  "waypoints": [{"outcome": "Demonstrate applied Machine learning", "target_horizon": "0-3 months",
                 "success_criteria": ["Complete two evaluated projects using it"]}],
  "assumptions": [], "risks": [], "confidence": 0.72,
  "needs_user_confirmation": [],
  "provenance": [{"source": "resume", "kind": "career_evidence"},
                 {"source": "onet", "kind": "role_benchmark", "role": "Data Scientists", "retrieved_at": "..."}],
  "created_at": "2026-11-24T12:00:00+00:00"
}
```
The payload is validated before forwarding; an invalid payload is never sent, and forwarding failures fall back to copy-ready JSON.

## Handoff behavior
- `AGENT_3_ADDRESS` unset → payload is copy-ready text only
- `AGENT_3_ADDRESS` set → payload is also sent to the Goal Planning agent; on delivery failure you still get the copy-ready JSON

## Environment variables
| Variable | Purpose |
| --- | --- |
| `ASI_ONE_API_KEY` | ASI:One chat completions (required) |
| `AGENTVERSE_API_KEY` | Agentverse mailbox registration (required) |
| `AGENT_SEED` | fixes the agent address (required, secret) |
| `ASI_ONE_MODEL` | `asi1` (default), `asi1-mini`, `asi1-ultra` |
| `ONET_API_KEY` | O*NET Web Services v2 key; without it analysis proceeds on the role name alone with reduced confidence |
| `AGENT_3_ADDRESS` | Goal Planning agent address for auto-forwarding |

## Data source — and its limits
Occupation grounding comes from [O*NET Web Services](https://services.onetcenter.org) (U.S. Department of Labor). **O*NET is primarily US-oriented**: job zones, outlook and education expectations describe the US labor market and are not calibrated elsewhere. Benchmarks record source, retrieval date and unavailable fields; occupational data is never fabricated when a request fails.

## Privacy limitations
- Anonymous profile ids for downstream payloads; no name/headline/location leaves this agent
- Logs carry trace ids and counts only — never raw résumé text or contact details (emails, phones, URLs are redacted at extraction)
- Résumé content is parsed in-process; nothing is persisted beyond the session key-value store, which holds sanitized facts only. Clear it by starting a new session; assume retention for the lifetime of the running agent process
- No protected characteristics are inferred or used in scoring
- Structured facts, user-provided facts, O*NET-sourced data and estimates are labeled by provenance so they stay distinguishable

## Example conversation
> **User:** `{"name": "Maya Chen", ... , "target_role": "Data Scientist"}` (alumni_history)
> **PathFinder:** Profile parsed — 7.3 years' experience, 4 skills with evidence, 0 conflicts. *(detail card)*
> **User:** Analyze my readiness.
> **PathFinder:** Readiness 66/100, medium difficulty — top gap: Machine learning (unverified). *(detail card)*
> **User:** What are my options?
> **PathFinder:** Three options *(carousel: skill-bridge ⭐, direct, stepping-stone — each with horizon and confidence)*
> **User:** *(taps Skill-bridge)*
> **PathFinder:** Phased waypoints, risks, scores *(trajectory card with Plan-my-goals CTA)*
> **User:** plan my goals
> **PathFinder:** Validated v2.0 payload, copy-ready and forwarded if `AGENT_3_ADDRESS` is set

## Known limitations
- Free-text parsing is heuristic: unusual formats may produce "undated role" warnings (by design — send structured JSON for exact results)
- Legacy `.doc` (binary Word) is not supported — save as `.docx` or paste text
- Salary, local-market and employer-specific factors are not modeled
- New agents can take a few minutes to be indexed by ASI:One search
- Confidence values reflect evidence quality, not the probability of any outcome
