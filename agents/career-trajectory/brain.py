"""Brain: a LangChain agent loop on the ASI:One chat completions API.

Stage-2 split of responsibilities: the deterministic engine (evidence,
benchmark, scoring, trajectories) lives in ``career_profile`` / ``resume`` /
``onet`` / ``scoring`` / ``trajectories``; this module is a thin orchestration
layer. Tools return JSON strings, never raise, and every number the model
shows comes from a tool — the LLM narrates, it never computes scores.

The agent is built lazily so importing this module never requires an API key
(and tests never touch the real ASI:One API).
"""

import json
import os
from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import career_profile as pf
import onet
import scoring
import trajectories

load_dotenv()

ASI_ONE_BASE_URL = "https://api.asi1.ai/v1"

_TODAY = date.today()

SYSTEM_PROMPT = """You are PathFinder, the career trajectory agent.
You turn career evidence (LinkedIn history, résumé, pasted experience) into an evidence-based
trajectory toward a target role. Flow:
1. Call parse_history on any history or résumé text the user sends.
2. Call lookup_occupation for the target role once known. If no target role is known, ask ONE
   short question and call no tools.
3. Call analyze_readiness with the stored profile and benchmark JSON from the session context,
   then plan_trajectories to get path options and show them.
4. When the user picks an option, call choose_path. For "what if" questions call what_if.
All scores, gaps and paths come from tools: never invent or recompute numbers, never invent
employers, titles, skills or dates. Surface conflicts and ask before assuming. Answer in 1-3
short sentences."""


def _loads(text, fallback):
    """Tolerant JSON dict parser for tool inputs the model relays from context."""
    if isinstance(text, dict):
        return text
    try:
        data = json.loads(text if isinstance(text, str) else str(text))
        return data if isinstance(data, dict) else fallback
    except (TypeError, ValueError):
        return fallback


def _facts_from_profile(profile: dict) -> dict:
    """Flat card-ready facts about the unified profile (legacy keys preserved)."""
    as_of = pf.date.fromisoformat(profile.get("as_of")) if profile.get("as_of") else _TODAY
    roles = profile.get("experiences") or []
    durations = [d for r in roles if (d := pf.duration_years(r.get("start"), r.get("end"), as_of)) is not None]
    years = sum(durations)
    current = next((r for r in roles if pf.is_present(r.get("end"))), None)
    if current is None and roles:
        current = max(roles, key=lambda r: pf.parse_month(r.get("end"), as_of) or (0, 0))
    skills = [s["name"] for s in profile.get("skills") or []]
    demonstrated = [s["name"] for s in profile.get("skills") or [] if s.get("tier") == "demonstrated"]
    progression = " -> ".join(r["title"] for r in roles if r.get("title")) or "not available"
    facts = {
        "Years of experience": f"{years:.1f}" if durations else "unknown",
        "Roles held": str(len(roles)),
        "Current role": (current.get("title") or "unknown") if current else "unknown",
        "Latest employer": pf.redact(current.get("company")) if current and current.get("company") else "unknown",
        "Average tenure": f"{years / len(durations):.1f} years" if durations else "unknown",
        "Top skills": "; ".join(skills[:8]) or "not listed",
        "Progression": progression,
        "Demonstrated evidence": "; ".join(demonstrated[:6]) or "none yet",
        "Profile completeness": f"{round(pf.completeness(profile) * 100)}%",
    }
    name = profile.get("name") or profile.get("headline") or "Career profile"
    title = f"{name} - {profile['headline']}" if profile.get("headline") else name
    return title, facts


@tool
def parse_history(raw: str) -> str:
    """Parse career-history input into a canonical profile with evidence and provenance.
    Accepts alumni_history JSON, canonical v2 profile JSON, or pasted career/résumé text.
    Returns profile facts, parsing warnings, and any conflicts needing user confirmation.
    Input: the raw career history (JSON object/string or free text)."""
    # Structured JSON is the LinkedIn integration's format (provenance linkedin);
    # hand-typed text is the user's own account (provenance user).
    provenance = pf.PROVENANCE_LINKEDIN if str(raw or "").lstrip().startswith(("{", "[")) \
        else pf.PROVENANCE_USER
    try:
        result = pf.normalize_input(raw, provenance=provenance, as_of=_TODAY)
    except ValueError as exc:
        return json.dumps({
            "error": (
                f"Could not read a career history ({exc}). Send your history as alumni_history JSON, "
                "a résumé, or text with roles and dates."
            )
        })
    profile = result["profile"]
    profile["target_roles"] = profile.get("target_roles") or []
    title, facts = _facts_from_profile(profile)
    payload: dict = {
        "title": title,
        "facts": facts,
        "profile": profile,
        "warnings": result["warnings"][:6],
    }
    payload["target_role"] = (profile.get("target_roles") or [None])[0]
    if profile.get("conflicts"):
        payload["conflicts"] = profile["conflicts"]
        payload["needs_user_confirmation"] = pf.needs_confirmation(profile)
    return json.dumps(payload)


@tool
def lookup_occupation(role: str) -> str:
    """Build a target-role benchmark from O*NET: normalized title, core skills with required
    levels, typical tasks, job zone, education expectations, related titles and outlook.
    Also records the retrieval date and any fields that were unavailable.
    Input: a job title such as 'Data Scientist'."""
    benchmark = onet.provider().benchmark(role)
    if benchmark.get("provider") == "none":
        return json.dumps({
            "title": benchmark["title"],
            "facts": {"Benchmark source": "none - ONET_API_KEY not configured; analysis will use "
                                          "the role name only and lower confidence"},
            "benchmark": benchmark,
            "warning": "O*NET unavailable; proceeding without occupation data.",
        })
    if benchmark.get("unavailable_fields") and not benchmark.get("onet_code"):
        return json.dumps({
            "title": role,
            "facts": {"error": "O*NET lookup failed; proceeding without occupation data."},
            "error": "O*NET lookup failed; proceeding without occupation data.",
            "benchmark": benchmark,
        })
    return json.dumps({
        "title": benchmark["title"],
        "facts": onet.facts_from_benchmark(benchmark),
        "benchmark": benchmark,
    })


def _unified_profile(profile_json, resume_json) -> tuple[dict | None, list]:
    """Coerce stored context into one canonical profile, merging a résumé fragment if present."""
    warnings: list = []
    profile = _loads(profile_json, None)
    if profile is None:
        return None, warnings
    if not profile.get("schema_version"):
        try:
            profile = pf.normalize_input(profile, provenance=pf.PROVENANCE_LINKEDIN, as_of=_TODAY)["profile"]
        except ValueError:
            return None, warnings
    else:
        profile = {**pf.empty_profile(profile.get("profile_id")), **profile}
    fragment = _loads(resume_json, None)
    if fragment:
        if not fragment.get("schema_version"):
            try:
                fragment = pf.normalize_input(fragment, provenance=pf.PROVENANCE_RESUME, as_of=_TODAY)["profile"]
            except ValueError:
                fragment = None
        if fragment:
            profile = pf.reconcile(profile, fragment)
    return profile, warnings


@tool
def analyze_readiness(profile_json: str, benchmark_json: str, resume_json: str = "") -> str:
    """Deterministically score readiness against a target role: readiness, transferability,
    evidence strength, transition difficulty, per-category breakdown, strengths, classified
    skill gaps (confirmed_gap vs missing_evidence vs unverified), assumptions and confidence.
    Input: the stored profile JSON, the stored benchmark JSON, and optionally a stored résumé
    fragment JSON to merge first (all from session context)."""
    profile, _ = _unified_profile(profile_json, resume_json)
    if profile is None:
        return json.dumps({"needs_input": {
            "field": "history", "label": "Share your career history (JSON or pasted text) first.",
        }, "message": "no profile on file yet"})
    benchmark = _loads(benchmark_json, None)
    target = (profile.get("target_roles") or [None])[0] or (benchmark or {}).get("role")
    if not target:
        return json.dumps({"needs_input": {
            "field": "target_role", "label": "What target role should the trajectory aim for?",
        }, "message": "target role missing"})
    if not benchmark or not benchmark.get("role"):
        benchmark = onet.EmptyBenchmark().benchmark(target)
    analysis = scoring.analyze(profile, benchmark, as_of=_TODAY)
    _, facts = _facts_from_profile(profile)
    return json.dumps({
        "title": f"Readiness for {target}",
        "facts": {**facts, **scoring.facts_from_analysis(analysis)},
        "analysis": analysis,
        "profile": profile,
        "benchmark": benchmark,
        "target_role": target,
    })


@tool
def plan_trajectories(analysis_json: str, profile_json: str, benchmark_json: str) -> str:
    """Generate 2-3 credible trajectory options (direct, skill-bridge, intermediate-role,
    longer reskilling), compare them, and recommend one. Each option carries waypoints,
    rationale, estimated horizon, risks, constraint fit and confidence.
    Input: the stored analysis, profile and benchmark JSON from session context."""
    data = _loads(analysis_json, {})
    profile = _loads(profile_json, {})
    benchmark = _loads(benchmark_json, {})
    if not data.get("readiness_score") and not data.get("target_role"):
        return json.dumps({"error": "run analyze_readiness first; pass its analysis JSON here."})
    plan_data = trajectories.plan(profile, data, benchmark)
    options = []
    for path in plan_data["paths"]:
        fit = path.get("constraint_fit") or {}
        subtitle = f"{path['type'].replace('_', ' ')} - about {path['estimated_months']} months, " \
                   f"confidence {path['confidence']:.0%}"
        if fit.get("timeline_ok") is False:
            subtitle += " (longer than your timeline)"
        options.append({
            "id": path["id"],
            "title": path["title"],
            "summary": subtitle[:200],
            "skill_gaps": path.get("required_skills", [])[:5],
        })
    return json.dumps({
        "title": plan_data["title"],
        "target_role": plan_data["target_role"],
        "paths": options,
        "plan": plan_data,
        "analysis": data,
        "profile": profile,
        "benchmark": benchmark,
        "recommended_id": plan_data["recommended_id"],
    })


@tool
def choose_path(plan_json: str, analysis_json: str, path_id: str) -> str:
    """Mark one trajectory option as selected and produce its detailed trajectory with
    waypoints, scores, risks and alternatives.
    Input: the stored plan JSON, the stored analysis JSON, and the chosen path id
    (for example 'path-1')."""
    plan_data = _loads(plan_json, {})
    analysis = _loads(analysis_json, {})
    if not plan_data.get("paths"):
        return json.dumps({"error": "run plan_trajectories first."})
    selection = trajectories.select(plan_data, analysis, str(path_id or "").strip())
    if "error" in selection:
        return json.dumps(selection)
    return json.dumps({
        **selection,
        "plan": plan_data,
        "analysis": analysis,
    })


@tool
def what_if(profile_json: str, benchmarks_json: str, scenarios_json: str) -> str:
    """Compare what-if scenarios against the current plan using the same scoring engine:
    relocation, remote-only roles, an added certification, a degree, an accelerated study
    schedule, a different timeline/industry, or different target roles.
    Input: the stored profile JSON; a JSON object mapping role name -> benchmark JSON
    (fetch each target role with lookup_occupation first); and a JSON list of scenarios like
    [{"type": "add_certification", "value": "CompTIA Data+"}, {"type": "alternate_role", "role": "Data Engineer"}]."""
    profile = _loads(profile_json, {})
    benchmarks_raw = _loads(benchmarks_json, {})
    scenarios = scenarios_json
    if isinstance(scenarios, str):
        try:
            scenarios = json.loads(scenarios)
        except ValueError:
            scenarios = []
    if isinstance(scenarios, dict):
        scenarios = scenarios.get("scenarios") or []
    if not isinstance(scenarios, list) or not scenarios:
        return json.dumps({"error": "pass a JSON list of scenarios, e.g. "
                                    '[{"type": "add_certification", "value": "CompTIA Data+"}].'})
    if not benchmarks_raw:
        return json.dumps({"error": "call lookup_occupation for the target role(s) first."})
    try:
        result = scoring.what_if(profile, benchmarks_raw, scenarios, as_of=_TODAY)
    except (KeyError, TypeError, ValueError) as exc:
        return json.dumps({"error": f"scenario comparison failed ({pf.redact(str(exc))[:80]})."})
    facts = {}
    for row in result["rows"]:
        readiness = "n/a" if row["readiness"] is None else f"{row['readiness']}/100"
        delta = "" if not row["delta"] else f" ({row['delta']:+d})"
        facts[str(row["scenario"])[:60]] = f"readiness {readiness}{delta}, difficulty {row['difficulty'] or 'unknown'}"
    return json.dumps({
        "title": "What-if comparison",
        "facts": facts,
        "what_if": result,
    })


_agent = None


def _get_agent():
    """Build the create_agent loop on first use (keeps import key-free for tests)."""
    global _agent
    if _agent is None:
        llm = ChatOpenAI(
            base_url=ASI_ONE_BASE_URL,
            api_key=os.environ["ASI_ONE_API_KEY"],
            model=os.getenv("ASI_ONE_MODEL", "asi1"),
            temperature=0,
            timeout=30,
        )
        _agent = create_agent(
            llm,
            tools=[parse_history, lookup_occupation, analyze_readiness, plan_trajectories,
                   choose_path, what_if],
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


async def answer(text: str) -> tuple[str, dict | None]:
    """Run the loop. Return (reply_text, last_tool_result_or_None)."""
    result = await _get_agent().ainvoke({"messages": [{"role": "user", "content": text}]})
    msgs = result["messages"]
    tool_data = None
    for message in reversed(msgs):
        if isinstance(message, ToolMessage):
            try:
                tool_data = json.loads(message.content)
            except (TypeError, ValueError):
                pass
            break
    return msgs[-1].content, tool_data
