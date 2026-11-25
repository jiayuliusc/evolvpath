"""Deterministic, configurable scoring engine, gap analysis, and what-if scenarios.

All scoring is computed from the normalized profile + target-role benchmark in
plain Python — the LLM never computes or inflates scores. Given identical
normalized inputs and the same ``as_of`` date, every number below is
reproducible. Scores are bounded 0-100 and are career-planning guidance, not
hiring eligibility or candidate ranking.

Default rubric (documented; adjust via ``ScoringConfig``):

- Relevant experience 30% - Skill coverage 25% - Achievement evidence 15%
- Seniority and scope 10% - Domain alignment 10%
- Education and certifications 5% - Recency 5%

Missing information never silently lowers a category: each category reports
its assumptions, and missing-data handling is documented per category below.
Evidence levels are weighted (demonstrated 1.0 > applied 0.7 > mentioned 0.4 >
inferred 0.15), and missing résumé evidence is NOT treated as missing real
ability: absence of evidence on a thin profile yields ``missing_evidence`` /
``unverified`` gap types rather than ``confirmed_gap``.
"""

from datetime import date
from typing import ClassVar

import career_profile as pf

# Documented proficiency scale: 0 none, 1 awareness, 2 working, 3 proficient, 4 expert.


class ScoringConfig:
    """Configurable knobs; defaults implement the documented rubric."""

    weights: ClassVar[dict] = {
        "relevant_experience": 0.30,
        "skill_coverage": 0.25,
        "achievement_evidence": 0.15,
        "seniority_scope": 0.10,
        "domain_alignment": 0.10,
        "education_certifications": 0.05,
        "recency": 0.05,
    }
    relevant_years_for_full_credit = 4.0  # relevant years at/above which the category caps
    total_years_for_full_credit = 6.0
    recency_grace_months = 6  # <=6 months since last activity scores 100
    recency_decay_per_month = 1.5  # linear decay to 0 at ~72 months
    recency_floor_months = 72
    missing_benchmark_neutral = 60  # neutral score when the benchmark lacks a category input
    missing_profile_neutral = 50  # neutral score when the profile lacks a category input
    confirmed_gap_completeness = 0.5  # completeness above which absence of evidence is a real gap
    difficulty_thresholds: ClassVar[dict] = {"low": 75, "medium": 55}  # readiness thresholds; else "high"
    confidence_floor, confidence_ceiling = 0.10, 0.95

    def __init__(self, **overrides):
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)


_SENIORITY_SIGNALS = (
    ("senior", 1), ("sr", 1), ("lead", 1), ("staff", 1), ("principal", 2),
    ("manager", 1), ("director", 2), ("vp", 2), ("head of", 2),
)
_STOPWORDS = {"the", "of", "and", "a", "an", "for", "in", "at", "to", "senior", "junior",
              "sr", "jr", "lead", "staff", "principal", "i", "ii", "iii"}


def _words(text: str) -> set[str]:
    return {w for w in pf.normalize_skill(text).replace(",", " ").split()
            if len(w) > 2 and w not in _STOPWORDS}


def _profile_years(profile: dict, as_of: date) -> tuple[float, int]:
    """Total dated years and count of undated roles."""
    years, undated = 0.0, 0
    for role in profile.get("experiences") or []:
        span = pf.duration_years(role.get("start"), role.get("end"), as_of)
        if span is None:
            undated += 1
        else:
            years += span
    return years, undated


def _match_benchmark_skills(profile: dict, benchmark: dict) -> list[dict]:
    """Pair each benchmark skill with the profile skill (if any) via key matching."""
    by_key = {s["key"]: s for s in profile.get("skills") or []}
    matches = []
    for required in benchmark.get("core_skills") or []:
        key = pf.normalize_skill(required.get("name"))
        match = by_key.get(key)
        if match is None:  # tolerant containment, e.g. "Machine Learning" vs "ML"
            for candidate_key, candidate in by_key.items():
                if len(key) >= 4 and (key in candidate_key or candidate_key in key):
                    match = candidate
                    break
        matches.append({
            "name": required.get("name"),
            "required_level": int(required.get("required_level") or 1),
            "importance": required.get("importance"),
            "profile": match,
        })
    return matches


def _relevant_years(profile: dict, benchmark: dict, as_of: date) -> float:
    """Years in roles whose titles or skills overlap the target-role family."""
    benchmark_words = _words(benchmark.get("title") or "")
    benchmark_skill_keys = {pf.normalize_skill(s.get("name")) for s in benchmark.get("core_skills") or []}
    relevant = 0.0
    for role in profile.get("experiences") or []:
        span = pf.duration_years(role.get("start"), role.get("end"), as_of)
        if span is None:
            continue
        title_overlap = bool(_words(role.get("title") or "") & benchmark_words)
        skill_overlap = bool(
            {pf.normalize_skill(s) for s in role.get("skills") or []} & benchmark_skill_keys
        )
        if title_overlap or skill_overlap:
            relevant += span
    return relevant


def _seniority_estimate(profile: dict, as_of: date) -> int:
    years, _ = _profile_years(profile, as_of)
    signals = 0
    for role in profile.get("experiences") or []:
        title = str(role.get("title") or "").lower()
        if is_managerish(title):
            signals += 1
        signals += sum(weight for word, weight in _SENIORITY_SIGNALS if word in title and word != "senior")
        if "senior" in title:
            signals += 1
    estimate = 1 + int(years // 5) + signals
    return max(1, min(estimate, 5))


def is_managerish(title: str) -> bool:
    return any(word in str(title or "").lower() for word in ("manager", "director", "head of", "vp"))


# --- categories -----------------------------------------------------------------


def _relevant_experience(profile, benchmark, as_of, config: ScoringConfig) -> dict:
    years, undated = _profile_years(profile, as_of)
    relevant = _relevant_years(profile, benchmark, as_of)
    assumptions = []
    if undated:
        assumptions.append(
            f"{undated} role(s) had no usable dates and contribute no experience years."
        )
    relevant_score = 100 * min(relevant / config.relevant_years_for_full_credit, 1.0)
    total_score = 100 * min(years / config.total_years_for_full_credit, 1.0)
    score = round(0.75 * relevant_score + 0.25 * total_score)
    evidence = [f"{relevant:.1f} of {years:.1f} total years align with the target role"]
    gaps = []
    if relevant == 0:
        gaps.append("No dated experience aligns with the target role family yet.")
    return {"score": max(0, min(score, 100)), "evidence": evidence, "gaps": gaps, "assumptions": assumptions}


def _skill_coverage(profile, benchmark, as_of, config: ScoringConfig, matches) -> dict:
    assumptions, evidence, gaps = [], [], []
    if not matches:
        assumptions.append(
            "Benchmark skill list unavailable; skill coverage scored neutral and confidence reduced."
        )
        return {"score": config.missing_benchmark_neutral, "evidence": evidence,
                "gaps": gaps, "assumptions": assumptions}
    total_weight, covered = 0.0, 0.0
    for match in matches:
        weight = max(match["required_level"], 1)
        total_weight += weight
        if match["profile"] is None:
            continue
        covered += weight * pf.EVIDENCE_WEIGHTS.get(match["profile"].get("tier"), 0.15)
        if match["profile"].get("tier") in ("demonstrated", "applied"):
            evidence.append(
                f"{match['name']}: {match['profile'].get('tier')} evidence "
                f"(level {match['profile'].get('level')}/4)"
            )
        else:
            gaps.append(f"{match['name']}: only {match['profile'].get('tier')} evidence for a "
                        f"level-{match['required_level']} requirement")
    score = round(100 * covered / total_weight) if total_weight else config.missing_benchmark_neutral
    return {"score": max(0, min(score, 100)), "evidence": evidence[:8], "gaps": gaps[:8],
            "assumptions": assumptions}


def _achievement_evidence(profile, benchmark, as_of, config: ScoringConfig, matches) -> dict:
    quantified = sum(
        1 for role in profile.get("experiences") or []
        for achievement in role.get("achievements") or [] if achievement.get("quantified")
    )
    demonstrated = sum(1 for m in matches if m["profile"] and m["profile"].get("tier") == "demonstrated")
    projects = len(profile.get("projects") or [])
    score = min(30 * min(quantified, 3) + 10 * min(demonstrated, 4) + 10 * min(projects, 2), 100)
    evidence = [f"{quantified} quantified achievement(s), {demonstrated} demonstrated benchmark skill(s)",
                f"{projects} project(s) on record"]
    gaps = []
    if quantified == 0 and demonstrated == 0:
        gaps.append("No quantified outcomes or demonstrated benchmark skills found in the evidence.")
    return {"score": score, "evidence": evidence, "gaps": gaps, "assumptions": []}


def _seniority_scope(profile, benchmark, as_of, config: ScoringConfig) -> dict:
    assumptions = []
    zone = benchmark.get("job_zone")
    estimate = _seniority_estimate(profile, as_of)
    if zone is None:
        assumptions.append("Job zone unavailable; seniority scored neutral and confidence reduced.")
        return {"score": config.missing_benchmark_neutral, "evidence": [f"Estimated seniority band {estimate}/5"],
                "gaps": [], "assumptions": assumptions}
    delta = max(0, int(zone) - estimate)
    score = round(100 * max(0.0, 1.0 - 0.25 * delta))
    evidence = [f"Estimated seniority band {estimate}/5 vs benchmark job zone {zone}/5"]
    gaps = [f"Target role typically sits in job zone {zone}; evidence suggests {estimate}."] if delta else []
    return {"score": score, "evidence": evidence, "gaps": gaps, "assumptions": assumptions}


def _domain_alignment(profile, benchmark, as_of, config: ScoringConfig) -> dict:
    user_text = " ".join(
        [profile.get("headline") or ""]
        + [str(role.get("title") or "") for role in profile.get("experiences") or []]
        + [str(p.get("name") or "") for p in profile.get("projects") or []]
        + [str(i) for i in (profile.get("preferences") or {}).get("industries") or []]
    )
    benchmark_text = " ".join(
        [benchmark.get("title") or "", benchmark.get("description") or ""]
        + list(benchmark.get("related_titles") or [])
    )
    user_words, benchmark_words = _words(user_text), _words(benchmark_text)
    if not benchmark_words or not user_words:
        return {"score": config.missing_profile_neutral,
                "evidence": [], "gaps": ["Not enough title/domain text to judge domain alignment."],
                "assumptions": ["Domain alignment scored neutral on sparse data."]}
    overlap = len(user_words & benchmark_words)
    exact = pf.normalize_skill(benchmark.get("title") or "") in pf.normalize_skill(user_text)
    score = round(100 * min(0.25 * overlap + (0.5 if exact else 0.0), 1.0))
    return {"score": max(0, min(score, 100)),
            "evidence": [f"{overlap} shared domain term(s)"],
            "gaps": [] if overlap else ["No vocabulary overlap with the target role family."],
            "assumptions": []}


_EDUCATION_RANK = {"associate": 1, "bachelor": 2, "bachelors": 2, "master": 3, "masters": 3,
                   "mba": 3, "doctorate": 4, "ph.d": 4, "phd": 4}


def _education_rank(text: str) -> int:
    lowered = str(text or "").lower()
    for token, rank in _EDUCATION_RANK.items():
        if token in lowered:
            return rank
    return 0


def _education_certifications(profile, benchmark, as_of, config: ScoringConfig) -> dict:
    assumptions = []
    required_text = benchmark.get("education")
    user_rank = max([_education_rank(e.get("degree")) for e in profile.get("education") or []] or [0])
    certs = len(profile.get("certifications") or [])
    if not required_text:
        assumptions.append(
            "Benchmark states no typical education requirement; education scored neutral "
            "(5% weight, certifications counted as a small bonus)."
        )
        score = config.missing_benchmark_neutral + min(4 * certs, 16)
        return {"score": min(score, 100), "evidence": [f"{certs} certification(s) on record"],
                "gaps": [], "assumptions": assumptions}
    required_rank = _education_rank(required_text) or 2  # O*NET defaults to bachelor-level wording
    if user_rank == 0:
        assumptions.append("No education entries found in the evidence; scored conservatively, not as inability.")
        score = config.missing_profile_neutral
        gaps = [f"Benchmark typically requires {required_text}; no education evidence on file."]
    elif user_rank >= required_rank:
        score = 100
        gaps = []
    else:
        score = 55
        gaps = [f"Benchmark typically requires {required_text}; strongest recorded education is lower."]
    score = min(score + min(4 * certs, 12), 100)
    return {"score": score, "evidence": [f"{certs} certification(s) on record"],
            "gaps": gaps, "assumptions": assumptions}


def _recency(profile, benchmark, as_of, config: ScoringConfig) -> dict:
    latest_month = None
    for role in profile.get("experiences") or []:
        end = pf.parse_month(role.get("end"), as_of)
        if end and (latest_month is None or end > latest_month):
            latest_month = end
    if latest_month is None:
        return {"score": config.missing_profile_neutral, "evidence": [],
                "gaps": ["No dated activity on record; recency unknown."],
                "assumptions": ["Recency scored neutral because no dates are available."]}
    months = max(0, (as_of.year - latest_month[0]) * 12 + as_of.month - latest_month[1])
    if months <= config.recency_grace_months:
        score = 100
    else:
        score = round(max(0.0, 100 - config.recency_decay_per_month *
                          (months - config.recency_grace_months)))
    label = "current role as of the evaluation date" if months <= config.recency_grace_months else \
        f"most recent dated activity {months} months before the evaluation date"
    return {"score": max(0, min(score, 100)), "evidence": [label], "gaps": [], "assumptions": []}


# --- gap analysis ---------------------------------------------------------------


def build_gaps(profile: dict, benchmark: dict, matches: list[dict], completeness: float) -> list[dict]:
    """Classify every shortfall benchmark skill as one of:

    - ``confirmed_gap`` — real evidence exists but sits below the required level,
      or the profile is detailed enough that absence of any mention is telling.
    - ``missing_evidence`` — nothing on file either way (thin profile); this is
      NOT treated as inability.
    - ``unverified`` — only mentioned/inferred support, too weak to count.
    """
    gaps = []
    # Absence of a skill is only treated as a confirmed gap when the profile is
    # detailed enough that absence is telling: solid completeness AND outcome
    # evidence (achievements/projects). Otherwise absence means missing evidence.
    has_outcome_evidence = bool(
        any(role.get("achievements") for role in profile.get("experiences") or [])
        or profile.get("projects")
    )
    detailed_profile = completeness >= ScoringConfig.confirmed_gap_completeness and has_outcome_evidence
    for match in matches:
        required = match["required_level"]
        current_skill = match["profile"]
        current_level = (current_skill or {}).get("level") or 0
        if current_level >= required:
            continue
        tier = (current_skill or {}).get("tier")
        evidence_items = (current_skill or {}).get("evidence") or []
        if current_level == 0 and not evidence_items:
            gap_type = "confirmed_gap" if detailed_profile else "missing_evidence"
        elif tier in ("mentioned", "inferred"):
            gap_type = "unverified"
        else:
            gap_type = "confirmed_gap"
        diff = required - current_level
        priority = "high" if (diff >= 2 or required >= 3) else ("medium" if diff == 1 else "low")
        gaps.append({
            "skill": match["name"],
            "current_level": current_level,
            "required_level": required,
            "gap_type": gap_type,
            "priority": priority,
            "evidence": [
                f"{item.get('tier')} via {item.get('source')}: {item.get('detail')}".strip(": ")
                for item in evidence_items[:2]
            ],
            "source": benchmark.get("provider") or "O*NET",
        })
    high_first = {"high": 0, "medium": 1, "low": 2}
    return sorted(gaps, key=lambda g: (high_first[g["priority"]], -g["required_level"]))[:12]


def _prerequisites(gaps: list[dict], benchmark: dict, profile: dict, as_of: date) -> dict:
    high = [g for g in gaps if g["priority"] == "high"]
    years, _ = _profile_years(profile, as_of)
    zone = benchmark.get("job_zone")
    estimate = _seniority_estimate(profile, as_of)
    return {
        "portfolio": [
            f"Build an evaluated project demonstrating {g['skill']}"
            for g in high[:3] if g["gap_type"] in ("confirmed_gap", "missing_evidence")
        ],
        "education": [] if benchmark.get("education") is None else (
            [] if profile.get("education") else
            [f"Benchmark typically lists {benchmark['education']}; confirm yours or plan a credential."]
        ),
        "experience": [] if zone is None or estimate >= zone else
        [f"Seniority gap: target role sits in job zone {zone}; current evidence suggests {estimate}."],
        "seniority": [f"{years:.1f} years of dated experience on record"] if years else [],
    }


def _clarifying_questions(profile: dict, matches: list[dict]) -> list[str]:
    questions = []
    undated = [r for r in profile.get("experiences") or [] if not pf.parse_month(r.get("start"))]
    if undated:
        questions.append(f"When did you start at {undated[0].get('company') or undated[0].get('title') or 'your latest role'}?")
    unverified = [m for m in matches if m["profile"] and m["profile"].get("tier") in ("mentioned", "inferred")]
    if unverified:
        questions.append(f"Have you actually used {unverified[0]['name']} in a project or job? Your résumé only lists it.")
    if not profile.get("preferences", {}).get("timeline_months"):
        questions.append("Do you have a target timeline in months for this move?")
    return questions[:3]


# --- main entry -----------------------------------------------------------------


def analyze(profile: dict, benchmark: dict, as_of: date | None = None,
            config: ScoringConfig | None = None) -> dict:
    """Score the unified profile against one benchmark. Deterministic and bounded."""
    config = config or ScoringConfig()
    as_of = as_of or date.today()
    matches = _match_benchmark_skills(profile, benchmark)
    full = pf.completeness(profile)
    categories = {
        "relevant_experience": _relevant_experience(profile, benchmark, as_of, config),
        "skill_coverage": _skill_coverage(profile, benchmark, as_of, config, matches),
        "achievement_evidence": _achievement_evidence(profile, benchmark, as_of, config, matches),
        "seniority_scope": _seniority_scope(profile, benchmark, as_of, config),
        "domain_alignment": _domain_alignment(profile, benchmark, as_of, config),
        "education_certifications": _education_certifications(profile, benchmark, as_of, config),
        "recency": _recency(profile, benchmark, as_of, config),
    }
    readiness = round(sum(
        config.weights[name] * data["score"] for name, data in categories.items()
    ))
    readiness = max(0, min(readiness, 100))

    strengths = [
        f"{m['name']} (level {m['profile']['level']}/4, {m['profile']['tier']})"
        for m in matches if m["profile"] and m["profile"].get("level", 0) >= max(m["required_level"] - 1, 1)
    ]
    transferable = [
        m["name"] for m in matches
        if m["profile"] and m["profile"].get("tier") in ("applied", "demonstrated")
        and m["profile"].get("level", 0) < m["required_level"]
    ]
    gaps = build_gaps(profile, benchmark, matches, full)
    prerequisites = _prerequisites(gaps, benchmark, profile, as_of)

    # transferability: requirement weight covered at applied-or-better, blended with domain fit
    applied_weight = sum(
        max(m["required_level"], 1) for m in matches
        if m["profile"] and m["profile"].get("tier") in ("applied", "demonstrated")
    )
    required_weight = sum(max(m["required_level"], 1) for m in matches) or None
    if required_weight:
        transferability = round(100 * (0.8 * applied_weight / required_weight +
                                       0.2 * categories["domain_alignment"]["score"] / 100))
    else:
        transferability = round(0.6 * categories["domain_alignment"]["score"] + 20)
        categories["skill_coverage"]["assumptions"].append(
            "Transferability estimated from domain fit only; benchmark skills unavailable."
        )
    transferability = max(0, min(transferability, 100))

    total_skills = len(profile.get("skills") or []) or 1
    demonstrated_ratio = len([s for s in profile.get("skills") or [] if s.get("tier") == "demonstrated"]) / total_skills
    evidenced_ratio = len([s for s in profile.get("skills") or [] if s.get("tier") in ("demonstrated", "applied")]) / total_skills
    quantified = sum(1 for r in profile.get("experiences") or []
                     for a in r.get("achievements") or [] if a.get("quantified"))
    evidence_strength = 100 * (0.45 * demonstrated_ratio + 0.35 * evidenced_ratio +
                               0.2 * min(quantified / 3, 1.0))
    evidence_strength -= 10 * min(len(profile.get("conflicts") or []), 3)
    evidence_strength = max(0, min(round(evidence_strength), 100))

    high_gaps = [g for g in gaps if g["priority"] == "high" and g["gap_type"] == "confirmed_gap"]
    if readiness >= config.difficulty_thresholds["low"] and not high_gaps:
        difficulty = "low"
    elif readiness >= config.difficulty_thresholds["medium"] and len(high_gaps) <= 2:
        difficulty = "medium"
    else:
        difficulty = "high"

    warnings = list(profile.get("warnings") or [])
    conflicts = list(profile.get("conflicts") or [])
    confidence = (
        0.35 + 0.45 * full + 0.15 * (benchmark.get("benchmark_quality") or 0.0)
        - 0.03 * min(len(warnings), 5) - 0.05 * min(len(conflicts), 3)
    )
    confidence = round(max(config.confidence_floor, min(confidence, config.confidence_ceiling)), 2)

    assumptions = [a for data in categories.values() for a in data["assumptions"]]
    if (benchmark.get("unavailable_fields") or []):
        assumptions.append(
            "Benchmark fields unavailable from O*NET: " + "; ".join(benchmark["unavailable_fields"][:6]) + "."
        )
    return {
        "target_role": benchmark.get("role") or benchmark.get("title"),
        "as_of": as_of.isoformat(),
        "readiness_score": readiness,
        "transferability_score": transferability,
        "evidence_strength_score": evidence_strength,
        "transition_difficulty": difficulty,
        "confidence": confidence,
        "completeness": full,
        "benchmark_quality": benchmark.get("benchmark_quality", 0.0),
        "breakdown": {
            name: {
                "score": data["score"],
                "weight": config.weights[name],
                "evidence": data["evidence"],
                "gaps": data["gaps"],
                "assumptions": data["assumptions"],
            }
            for name, data in categories.items()
        },
        "strengths": strengths[:10],
        "transferable_skills": transferable[:10],
        "gaps": gaps,
        "prerequisites": prerequisites,
        "assumptions": assumptions,
        "warnings": warnings,
        "conflicts": conflicts,
        "needs_user_confirmation": pf.needs_confirmation(profile),
        "clarifying_questions": _clarifying_questions(profile, matches),
        "benchmark": {
            "role": benchmark.get("role"), "title": benchmark.get("title"),
            "provider": benchmark.get("provider"), "retrieved_at": benchmark.get("retrieved_at"),
            "unavailable_fields": benchmark.get("unavailable_fields") or [],
        },
    }


# --- what-if scenarios -----------------------------------------------------------

# Scenario modifiers reuse the SAME normalized profile and scoring engine; each
# only changes the inputs a real change would change. Effects are deliberately
# conservative and documented in the notes.


def apply_scenario(profile: dict, scenario: dict) -> tuple[dict, str]:
    """Return a modified copy of the profile plus a human note."""
    modified = {**profile, "preferences": dict(profile.get("preferences") or {})}
    kind = str(scenario.get("type") or scenario.get("scenario") or "").strip().lower()
    value = scenario.get("value", scenario.get("role", scenario.get("location", "")))
    notes = []
    if kind in ("add_certification", "certification"):
        name = str(value or "New certification")
        modified["certifications"] = list(profile.get("certifications") or []) + [
            {"name": name, "issuer": str(scenario.get("issuer") or ""), "year": "", "provenance": ["user"]}
        ]
        notes.append(f"Adds certification '{name}' (education/certification category only, 5% weight).")
    elif kind == "degree":
        degree = str(value or "Master's degree")
        modified["education"] = list(profile.get("education") or []) + [
            {"degree": degree, "field": str(scenario.get("field") or ""), "school": "", "end": "",
             "provenance": ["user"]}
        ]
        notes.append(f"Adds education entry '{degree}' (5% weight; only helps if the benchmark states it).")
    elif kind in ("remote_only", "remote"):
        modified["preferences"]["remote_preference"] = "remote"
        notes.append("Restricts to remote roles; affects constraint fit of paths, not scores.")
    elif kind in ("relocation", "relocate"):
        modified["preferences"]["location"] = str(value or "new location")
        notes.append(f"Relocation to '{value or 'new location'}'; O*NET data is US-national, "
                     "so local salary/market effects are not modeled.")
    elif kind in ("accelerated", "accelerated_schedule"):
        hours = modified["preferences"].get("available_hours_per_week") or 10
        modified["preferences"]["available_hours_per_week"] = max(int(hours), 20)
        notes.append("Assumes 20+ study hours/week; trajectory horizons shrink only in constraint fit.")
    elif kind in ("timeline", "timeline_months"):
        modified["preferences"]["timeline_months"] = int(value or 12)
        notes.append(f"Timeline set to {value} months; affects path constraint fit.")
    elif kind in ("industry", "target_industry"):
        industries = list(modified["preferences"].get("industries") or []) + [str(value)]
        modified["preferences"]["industries"] = industries
        notes.append(f"Targets the {value} industry; domain-alignment vocabulary only.")
    else:
        notes.append(f"Unknown scenario type '{kind}' - ignored.")
    return modified, " ".join(notes)


def what_if(profile: dict, benchmarks: dict[str, dict], scenarios: list[dict],
            as_of: date | None = None, config: ScoringConfig | None = None) -> dict:
    """Compare scenarios against the base case, reusing the same engine.

    ``benchmarks`` maps role name -> benchmark dict (``alternate_role`` scenarios
    must have their benchmark fetched by the caller first).
    """
    as_of = as_of or date.today()
    base_role = next(iter(benchmarks))
    base = analyze(profile, benchmarks[base_role], as_of, config)
    rows = [{
        "scenario": "current plan (baseline)",
        "target_role": base["target_role"],
        "readiness": base["readiness_score"],
        "difficulty": base["transition_difficulty"],
        "delta": 0,
        "notes": "",
    }]
    for scenario in scenarios:
        kind = str(scenario.get("type") or scenario.get("scenario") or "").lower()
        if kind in ("alternate_role", "different_target_role", "role"):
            role = str(scenario.get("role") or scenario.get("value") or "")
            benchmark = benchmarks.get(role)
            if benchmark is None:
                rows.append({"scenario": f"target {role}", "target_role": role, "readiness": None,
                             "difficulty": None, "delta": None,
                             "notes": "No benchmark fetched for this role; ask for it first."})
                continue
            result = analyze(profile, benchmark, as_of, config)
            notes = "Scores recomputed against a different benchmark."
        elif kind in ("different_industry", "industry", "target_industry"):
            modified, note = apply_scenario(profile, scenario)
            result = analyze(modified, benchmarks[base_role], as_of, config)
            notes = note
        else:
            modified, note = apply_scenario(profile, scenario)
            result = analyze(modified, benchmarks[base_role], as_of, config)
            notes = note
        rows.append({
            "scenario": str(scenario.get("label") or kind or "scenario"),
            "target_role": result["target_role"],
            "readiness": result["readiness_score"],
            "difficulty": result["transition_difficulty"],
            "delta": result["readiness_score"] - base["readiness_score"],
            "notes": notes,
        })
    return {"baseline_role": base_role, "as_of": as_of.isoformat(), "rows": rows,
            "disclaimer": "Guidance only; no outcome is guaranteed."}


def facts_from_analysis(analysis: dict) -> dict:
    """Flat, card-ready rows summarizing an analysis (bounded; no raw evidence)."""
    facts = {
        "Readiness": f"{analysis['readiness_score']}/100",
        "Transferability": f"{analysis['transferability_score']}/100",
        "Evidence strength": f"{analysis['evidence_strength_score']}/100",
        "Transition difficulty": analysis["transition_difficulty"],
        "Confidence": str(analysis["confidence"]),
        "Target role": analysis.get("target_role") or "not set",
    }
    high_gaps = [g for g in analysis.get("gaps") or [] if g["priority"] == "high"]
    if high_gaps:
        facts["Priority gaps"] = "; ".join(f"{g['skill']} ({g['gap_type']})" for g in high_gaps[:5])
    strengths = analysis.get("strengths") or []
    if strengths:
        facts["Top strengths"] = "; ".join(strengths[:5])
    if analysis.get("needs_user_confirmation"):
        facts["Needs confirmation"] = "; ".join(analysis["needs_user_confirmation"][:3])
    if analysis.get("assumptions"):
        facts["Assumptions"] = "; ".join(analysis["assumptions"][:4])
    return facts
