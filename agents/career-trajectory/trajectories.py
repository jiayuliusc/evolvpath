"""Career trajectory generation, comparison, and the versioned handoff payload.

``plan`` turns one analysis into 2-3 credible path options (direct transition,
skill-bridge, intermediate-role, longer reskilling) with ordered waypoints,
rationale, risks, constraint fit and confidence — all derived deterministically
from the profile, benchmark and scores. ``recommend`` compares options on
existing-skill fit, difficulty, time, gap severity, constraint fit and evidence
confidence. ``build_payload`` produces the versioned Goal Planning Agent payload
(schema 2.0); ``validate_payload`` gates forwarding.

PathFinder defines outcomes, gaps and waypoints only — never daily or weekly
schedules (that is the Goal Planning Agent's job), and nothing here guarantees
employment, salary, promotion, or completion dates.
"""

import hashlib
import json
from datetime import UTC, datetime

SCHEMA_VERSION = "2.0"

PATH_TYPE_DIRECT = "direct_transition"
PATH_TYPE_BRIDGE = "skill_bridge_transition"
PATH_TYPE_INTERMEDIATE = "intermediate_role_transition"
PATH_TYPE_RESKILL = "longer_reskilling_path"

# composite scoring weights for comparing options (documented)
COMPOSITE_WEIGHTS = {
    "skill_fit": 0.35,      # existing-skill fit (readiness)
    "ease": 0.15,           # transition difficulty, inverted
    "gaps": 0.15,           # number and severity of gaps
    "time": 0.10,           # shorter is better
    "constraints": 0.15,    # user-constraint fit
    "confidence": 0.10,     # evidence confidence
}

_DIFFICULTY_NUM = {"low": 10, "medium": 50, "high": 90}
_MONTH_CAP = 36
_MONTH_FLOOR = 3


def _gap_months(gaps: list[dict]) -> int:
    months = 0.0
    for gap in gaps:
        weight = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(gap.get("priority"), 1.0)
        # missing_evidence gaps are cheaper to verify than to close
        if gap.get("gap_type") == "missing_evidence":
            weight *= 0.5
        months += weight
    return int(min(max(months, 0), 18))


def _estimate_months(analysis: dict, path_type: str) -> int:
    gaps = analysis.get("gaps") or []
    months = _gap_months(gaps)
    prereq = analysis.get("prerequisites") or {}
    months += 2 * min(len(prereq.get("portfolio") or []), 2)
    months += 6 if any("Seniority gap" in item for item in prereq.get("experience") or []) else 0
    months += 12 if any("credential" in item or "confirm yours" in item
                        for item in prereq.get("education") or []) else 0
    if path_type == PATH_TYPE_INTERMEDIATE:
        months += 8  # time in the intermediate role before the final move
    elif path_type == PATH_TYPE_RESKILL:
        months += 12
    elif path_type == PATH_TYPE_DIRECT:
        months = max(months, 3)
    return int(min(max(months, _MONTH_FLOOR), _MONTH_CAP))


def _intermediate_role(analysis: dict, benchmark: dict, profile: dict) -> str:
    """Suggested stepping-stone title: a real O*NET related title sharing
    vocabulary with the user's history, else a clearly-labeled entry variant.
    Heuristic waypoint naming, not occupational data."""
    target = benchmark.get("title") or analysis.get("target_role") or "the target role"
    user_words = set()
    for role in profile.get("experiences") or []:
        user_words.update(word for word in str(role.get("title") or "").lower().split() if len(word) > 3)
    for related in benchmark.get("related_titles") or []:
        words = {w.lower() for w in related.split() if len(w) > 3}
        if words & user_words and related != target:
            return related
    return f"Entry-level {target} role (stepping stone)"


def _roles_for(path_type: str, profile: dict, analysis: dict, benchmark: dict) -> list[str]:
    target = benchmark.get("title") or analysis.get("target_role") or "Target role"
    current = (profile.get("experiences") or [{}])[-1].get("title") if profile.get("experiences") else ""
    roles = [r for r in (current, target) if r]
    if path_type in (PATH_TYPE_INTERMEDIATE, PATH_TYPE_RESKILL):
        roles = [r for r in (current, _intermediate_role(analysis, benchmark, profile), target) if r]
    return roles


def _capability_waypoints(analysis: dict, path_type: str, months: int) -> list[dict]:
    """Outcome-first waypoints for the Goal Planning Agent (0-4 levels of detail).

    Horizons partition the estimated window; success criteria are evidence
    checks, never schedules."""
    gaps = analysis.get("gaps") or []
    ordered = sorted(gaps, key=lambda g: {"high": 0, "medium": 1, "low": 2}.get(g["priority"], 2))[:5]
    if not ordered:
        ordered = [{"skill": "Applied practice in the target role", "priority": "medium", "gap_type": "missing_evidence"}]
    per = max(months // max(len(ordered), 1), 2)
    waypoints = []
    start = 0
    for i, gap in enumerate(ordered):
        end = min(start + per, months) if i < len(ordered) - 1 else months
        horizon = f"{start}-{end} months"
        start = end
        skill = gap.get("skill") or "target capability"
        if gap.get("gap_type") == "missing_evidence":
            outcome = f"Verify {skill} against real work (evidence check)"
            criteria = ["Collect one concrete artifact (project, ticket, or work sample) using it"]
        else:
            outcome = f"Demonstrate applied {skill}"
            criteria = ["Complete two evaluated projects using it",
                        "Get feedback from someone working in the target role"]
        waypoints.append({"outcome": outcome[:200], "target_horizon": horizon, "success_criteria": criteria})
    if path_type in (PATH_TYPE_INTERMEDIATE, PATH_TYPE_RESKILL):
        waypoints.append({"outcome": "Secure the stepping-stone role and perform in it for one review cycle",
                          "target_horizon": f"{months - 6}-{months} months" if months > 8 else "final stretch",
                          "success_criteria": ["Update evidence log with two outcomes from the new role"]})
    return waypoints


def _constraint_fit(analysis: dict, preferences: dict, months: int) -> dict:
    timeline = preferences.get("timeline_months")
    hours_available = preferences.get("available_hours_per_week")
    gaps = analysis.get("gaps") or []
    hours_needed = max(4, 2.5 * len([g for g in gaps if g["priority"] in ("high", "medium")]))
    timeline_ok = None if timeline is None else bool(timeline >= months)
    hours_ok = None if hours_available is None else bool(hours_available >= hours_needed)
    score = 100
    score -= 35 if timeline_ok is False else (15 if timeline_ok is None else 0)
    score -= 35 if hours_ok is False else (15 if hours_ok is None else 0)
    remote = (preferences.get("remote_preference") or "").lower()
    if remote == "remote":
        score -= 10  # remote-only narrows openings; O*NET data is US-national
    return {
        "score": max(0, min(score, 100)),
        "timeline_ok": timeline_ok,
        "hours_ok": hours_ok,
        "hours_needed_per_week": round(hours_needed, 1),
        "notes": [],
    }


def _make_path(path_type: str, profile: dict, analysis: dict, benchmark: dict, preferences: dict,
               index: int) -> dict:
    target = benchmark.get("title") or analysis.get("target_role") or "the target role"
    months = _estimate_months(analysis, path_type)
    readiness = analysis.get("readiness_score", 0)
    high_gaps = [g for g in analysis.get("gaps") or [] if g["priority"] == "high"]
    roles = _roles_for(path_type, profile, analysis, benchmark)
    titles = {
        PATH_TYPE_DIRECT: f"Direct transition to {target}",
        PATH_TYPE_BRIDGE: f"Skill-bridge into {target}",
        PATH_TYPE_INTERMEDIATE: f"Stepping-stone route through {_intermediate_role(analysis, benchmark, profile)}",
        PATH_TYPE_RESKILL: f"Longer reskilling path to {target}",
    }
    rationales = {
        PATH_TYPE_DIRECT:
            f"Readiness {readiness}/100 with no confirmed high-priority gaps: the evidence already supports "
            f"applying for {target} roles now.",
        PATH_TYPE_BRIDGE:
            f"Readiness {readiness}/100: existing evidence covers the core, and closing "
            f"{len(high_gaps)} high-priority gap(s) first makes the move credible.",
        PATH_TYPE_INTERMEDIATE:
            f"Readiness {readiness}/100 and a seniority/skill jump remain: a stepping-stone role builds the "
            f"missing evidence while staying employed.",
        PATH_TYPE_RESKILL:
            f"Readiness {readiness}/100 with substantial confirmed gaps: a structured reskilling phase "
            f"comes before any role move.",
    }
    fit = _constraint_fit(analysis, preferences, months)
    confidence = max(0.05, min(0.95, analysis.get("confidence", 0.5) + {
        PATH_TYPE_DIRECT: 0.05, PATH_TYPE_BRIDGE: 0.02,
        PATH_TYPE_INTERMEDIATE: 0.0, PATH_TYPE_RESKILL: -0.05,
    }[path_type]))
    strengths = analysis.get("strengths") or []
    advantages = {
        PATH_TYPE_DIRECT: ["Fastest route if applications succeed", "Builds on demonstrated evidence"],
        PATH_TYPE_BRIDGE: ["Keeps current income while closing gaps", "Targets only the missing skills"],
        PATH_TYPE_INTERMEDIATE: ["Lower risk per step", "Each waypoint generates fresh, verifiable evidence"],
        PATH_TYPE_RESKILL: ["Most thorough preparation", "Best fit when foundations are missing"],
    }[path_type]
    tradeoffs = {
        PATH_TYPE_DIRECT: ["Least preparation margin", "Assumes hiring cycles cooperate"],
        PATH_TYPE_BRIDGE: ["Requires sustained study alongside work"],
        PATH_TYPE_INTERMEDIATE: ["Longer total horizon", "Intermediate role may not exist internally"],
        PATH_TYPE_RESKILL: ["Longest horizon and highest opportunity cost"],
    }[path_type]
    return {
        "id": f"path-{index}",
        "title": titles[path_type][:120],
        "type": path_type,
        "roles": roles,
        "estimated_months": months,
        "rationale": rationales[path_type],
        "required_skills": [g["skill"] for g in (analysis.get("gaps") or [])[:6]],
        "evidence_support": strengths[:4],
        "risks": [
            *(["High-priority gaps: " + ", ".join(g["skill"] for g in high_gaps[:4])] if high_gaps else []),
            *(["Timeline preference is shorter than the estimate"] if fit["timeline_ok"] is False else []),
            *(["Weekly study demand exceeds stated availability"] if fit["hours_ok"] is False else []),
            "Estimates are planning guidance; no employment or completion dates are guaranteed.",
        ][:6],
        "prerequisites": [
            *(analysis.get("prerequisites") or {}).get("portfolio", [])[:2],
            *(analysis.get("prerequisites") or {}).get("experience", [])[:1],
        ],
        "constraint_fit": fit,
        "confidence": round(confidence, 2),
        "advantages": advantages,
        "tradeoffs": tradeoffs,
        "waypoints": _capability_waypoints(analysis, path_type, months),
    }


def _types_for(readiness: int, completeness: float) -> list[str]:
    """Which path archetypes the evidence supports, in priority order."""
    if readiness >= 75:
        return [PATH_TYPE_DIRECT, PATH_TYPE_BRIDGE, PATH_TYPE_INTERMEDIATE]
    if readiness >= 55:
        return [PATH_TYPE_BRIDGE, PATH_TYPE_DIRECT if readiness >= 65 else PATH_TYPE_INTERMEDIATE,
                PATH_TYPE_INTERMEDIATE]
    if readiness >= 35:
        return [PATH_TYPE_INTERMEDIATE, PATH_TYPE_BRIDGE, PATH_TYPE_RESKILL]
    return [PATH_TYPE_RESKILL, PATH_TYPE_INTERMEDIATE]


def _dedupe(types: list[str]) -> list[str]:
    seen, ordered = set(), []
    for path_type in types:
        if path_type not in seen:
            seen.add(path_type)
            ordered.append(path_type)
    return ordered


def plan(profile: dict, analysis: dict, benchmark: dict) -> dict:
    """Generate and compare credible trajectory options from one analysis."""
    preferences = profile.get("preferences") or {}
    readiness = analysis.get("readiness_score", 0)
    types = _dedupe(_types_for(readiness, analysis.get("completeness", 0)))
    paths = [_make_path(path_type, profile, analysis, benchmark, preferences, i + 1)
             for i, path_type in enumerate(types[:3])]
    comparison = [compare_row(path, analysis) for path in paths]
    recommended = max(comparison, key=lambda row: row["composite"])["id"]
    return {
        "title": f"Trajectory options toward {analysis.get('target_role') or 'your target role'}",
        "target_role": analysis.get("target_role"),
        "paths": paths,
        "comparison": comparison,
        "recommended_id": recommended,
        "disclaimer": "Career-planning guidance only; outcomes are not guaranteed.",
    }


def compare_row(path: dict, analysis: dict) -> dict:
    """One comparison row + documented composite score."""
    fit = path.get("constraint_fit") or {}
    months = path.get("estimated_months", 12)
    high = len([g for g in analysis.get("gaps") or [] if g["priority"] == "high"])
    medium = len([g for g in analysis.get("gaps") or [] if g["priority"] == "medium"])
    severity = max(0, 100 - 12 * high - 6 * medium - 2 * len(analysis.get("gaps") or []))
    parts = {
        "skill_fit": analysis.get("readiness_score", 0),
        "ease": 100 - _DIFFICULTY_NUM.get(analysis.get("transition_difficulty"), 50),
        "gaps": severity,
        "time": 100 * (1 - min(months / _MONTH_CAP, 1.0)),
        "constraints": fit.get("score", 50),
        "confidence": 100 * path.get("confidence", 0.5),
    }
    composite = round(sum(COMPOSITE_WEIGHTS[key] * value for key, value in parts.items()))
    return {
        "id": path["id"],
        "title": path["title"],
        "type": path["type"],
        "estimated_months": months,
        "high_priority_gaps": high + medium,
        "constraint_fit": fit.get("score", 50),
        "confidence": path.get("confidence"),
        "composite": max(0, min(composite, 100)),
    }


def select(plan_data: dict, analysis: dict, path_id: str) -> dict:
    """Mark one option selected and shape it for the trajectory card."""
    path = next((p for p in plan_data.get("paths") or [] if p.get("id") == path_id), None)
    if path is None:
        return {"error": f"unknown path '{path_id}'; pick one of the offered options."}
    return {
        "title": path["title"],
        "target_role": plan_data.get("target_role"),
        "selected": path_id,
        "type": path.get("type"),
        "scores": {
            "readiness": analysis.get("readiness_score"),
            "transferability": analysis.get("transferability_score"),
            "evidence_strength": analysis.get("evidence_strength_score"),
            "transition_difficulty": analysis.get("transition_difficulty"),
        },
        "trajectory": {
            "title": path["title"],
            "summary": path["rationale"],
            "roles": path["roles"],
            "estimated_months": path["estimated_months"],
            "phases": [
                {"n": i, "goal": w["outcome"], "horizon": w["target_horizon"],
                 "skills_to_build": w.get("success_criteria", [])}
                for i, w in enumerate(path.get("waypoints") or [], start=1)
            ],
            "skill_gaps": [g["skill"] for g in (analysis.get("gaps") or [])[:6]],
            "risks": path.get("risks") or [],
            "confidence": path.get("confidence"),
        },
        "alternatives": [
            {"id": p["id"], "title": p["title"], "type": p["type"],
             "estimated_months": p["estimated_months"]}
            for p in plan_data.get("paths") or [] if p.get("id") != path_id
        ],
    }


# --- Goal Planning Agent payload --------------------------------------------------


def _trajectory_id(profile_id: str, target: str, path_id: str) -> str:
    digest = hashlib.sha1(f"{profile_id}|{target}|{path_id}".encode()).hexdigest()[:10]
    return f"traj_{digest}"


def build_payload(profile: dict, analysis: dict, selection: dict) -> dict:
    """Versioned v2.0 payload for the Goal Planning Agent.

    Contains NO name/headline/location — only the anonymous profile_id — and no
    schedules: PathFinder owns outcomes, gaps and waypoints only.
    """
    target = selection.get("target_role") or analysis.get("target_role") or ""
    trajectory = selection.get("trajectory") or {}
    scores = selection.get("scores") or {}
    provenance = [{"source": source, "kind": "career_evidence"}
                  for source in sorted({item.get("source")
                                        for skill in profile.get("skills") or []
                                        for item in skill.get("evidence") or []
                                        if item.get("source")})]
    provenance.append({
        "source": "onet",
        "kind": "role_benchmark",
        "role": (analysis.get("benchmark") or {}).get("title") or target,
        "retrieved_at": (analysis.get("benchmark") or {}).get("retrieved_at"),
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": _trajectory_id(str(profile.get("profile_id")), target,
                                         str(selection.get("selected"))),
        "profile_id": profile.get("profile_id"),
        "trace_id": f"trace_{hashlib.sha1(str(profile.get('profile_id')).encode()).hexdigest()[:8]}",
        "target_role": target,
        "scores": {
            "readiness": scores.get("readiness"),
            "transferability": scores.get("transferability"),
            "evidence_strength": scores.get("evidence_strength"),
            "transition_difficulty": scores.get("transition_difficulty"),
        },
        "selected_path": {
            "title": trajectory.get("title") or selection.get("title") or "",
            "roles": trajectory.get("roles") or [],
            "estimated_months": trajectory.get("estimated_months"),
            "type": selection.get("type"),
            "rationale": trajectory.get("summary") or "",
            "confidence": trajectory.get("confidence"),
        },
        "alternatives": selection.get("alternatives") or [],
        "strengths": analysis.get("strengths") or [],
        "skill_gaps": analysis.get("gaps") or [],
        "waypoints": [
            {"outcome": phase.get("goal") or "", "target_horizon": phase.get("horizon") or "",
             "success_criteria": phase.get("skills_to_build") or []}
            for phase in trajectory.get("phases") or []
        ],
        "assumptions": analysis.get("assumptions") or [],
        "risks": trajectory.get("risks") or [],
        "confidence": analysis.get("confidence"),
        "needs_user_confirmation": analysis.get("needs_user_confirmation") or [],
        "provenance": provenance,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


REQUIRED_KEYS = ("schema_version", "trajectory_id", "profile_id", "target_role", "scores",
                 "selected_path", "waypoints", "confidence", "created_at")
_BANNED_KEY_FRAGMENTS = ("daily", "weekly", "schedule")
_GAP_TYPES = ("confirmed_gap", "missing_evidence", "unverified")


def validate_payload(payload: dict) -> list[str]:
    """Structural validation gating the Goal Planning handoff; empty list = valid."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload is not an object"]
    for key in REQUIRED_KEYS:
        if key not in payload or payload[key] in (None, "", [], {}):
            errors.append(f"missing required field '{key}'")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be '{SCHEMA_VERSION}'")
    if not str(payload.get("trajectory_id") or "").startswith("traj_"):
        errors.append("trajectory_id must start with 'traj_'")
    scores = payload.get("scores") or {}
    for key in ("readiness", "transferability", "evidence_strength"):
        value = scores.get(key)
        if not isinstance(value, int) or not 0 <= value <= 100:
            errors.append(f"scores.{key} must be an integer 0-100")
    if scores.get("transition_difficulty") not in ("low", "medium", "high"):
        errors.append("scores.transition_difficulty must be low|medium|high")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        errors.append("confidence must be 0-1")
    try:
        datetime.fromisoformat(str(payload.get("created_at")))
    except ValueError:
        errors.append("created_at must be ISO-8601")
    for key in payload:
        if any(fragment in str(key).lower() for fragment in _BANNED_KEY_FRAGMENTS):
            errors.append(f"field '{key}' looks like a schedule; PathFinder must not emit schedules")
    waypoints = payload.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        errors.append("waypoints must be a non-empty list")
    else:
        for i, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, dict) or not waypoint.get("outcome") \
                    or not waypoint.get("target_horizon"):
                errors.append(f"waypoint {i + 1} needs at least 'outcome' and 'target_horizon'")
            for key in waypoint or {}:
                if any(fragment in str(key).lower() for fragment in _BANNED_KEY_FRAGMENTS):
                    errors.append(f"waypoint {i + 1} field '{key}' looks like a schedule")
    for i, gap in enumerate(payload.get("skill_gaps") or []):
        if not isinstance(gap, dict) or not gap.get("skill") \
                or gap.get("gap_type") not in _GAP_TYPES:
            errors.append(f"skill_gaps[{i}] needs 'skill' and a valid gap_type")
            break
    return errors


def dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2)
