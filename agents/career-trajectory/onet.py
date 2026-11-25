"""Target-role benchmarking behind a provider interface.

``BenchmarkProvider`` is the extension point for additional regional or
labor-market data sources; ``OnetProvider`` implements it against O*NET Web
Services v2 (US Department of Labor). A benchmark records its source,
retrieval timestamp, and any fields that could not be retrieved
(``unavailable_fields``) — occupational data is never fabricated when an
external request fails; callers fall back to :class:`EmptyBenchmark`.

Importantly, benchmark skills are *requirements of the target role*, never
claims about the user: profile evidence and benchmark data are combined only
inside the scoring engine.
"""

import json
import os
from datetime import UTC, datetime

import httpx

ONET_BASE_URL = "https://api-v2.onetcenter.org"
HTTP_TIMEOUT = 10  # seconds; AGENTS.md caps every HTTP call at 10
SOURCE = "O*NET Web Services v2 (https://services.onetcenter.org)"

# O*NET v2 reports importance/level on 0-100 in some deployments and 1-5 in
# others; normalize defensively to the documented 0-100 scale.
def _scale(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= number <= 5 and number > 0:
        number *= 20
    return round(min(max(number, 0.0), 100.0), 1)


def importance_to_level(value: float | None) -> int:
    """Documented mapping from benchmark importance (0-100, or legacy 1-5 which
    is normalized first) to required level 0-4:
    >=85 expert, >=70 proficient, >=50 working, >=30 awareness, else optional."""
    scaled = _scale(value)
    if scaled is None:
        return 1
    if scaled >= 85:
        return 4
    if scaled >= 70:
        return 3
    if scaled >= 50:
        return 2
    if scaled >= 30:
        return 1
    return 0


class BenchmarkProvider:
    """Interface for occupational-data providers.

    Implementations return a benchmark dict with at least:
    ``role``, ``provider``, ``source``, ``retrieved_at``, ``title``,
    ``core_skills`` (``[{name, importance, required_level}]``),
    ``tasks``, ``education``, ``job_zone``, ``related_titles``, ``outlook``,
    ``description``, ``unavailable_fields``. Failures degrade to fewer fields
    plus ``unavailable_fields`` entries — never fabricated data.
    """

    def benchmark(self, role: str) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class EmptyBenchmark:
    """Null-object benchmark used when O*NET is unconfigured or unreachable.

    Skill requirements fall back to the requested role name only; every
    normally-sourced field is listed in ``unavailable_fields`` so downstream
    confidence and gaps reflect the missing data instead of pretending.
    """

    UNAVAILABLE = ("core_skills", "tasks", "education", "job_zone", "related_titles", "outlook", "description")

    def benchmark(self, role: str) -> dict:
        return {
            "role": role,
            "provider": "none",
            "source": None,
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "title": role,
            "onet_code": None,
            "description": "",
            "core_skills": [],
            "tasks": [],
            "education": None,
            "job_zone": None,
            "related_titles": [],
            "outlook": None,
            "unavailable_fields": list(self.UNAVAILABLE),
            "benchmark_quality": 0.0,
        }


class OnetProvider(BenchmarkProvider):
    """O*NET Web Services v2 benchmark provider (US-oriented data).

    Auth uses the ``X-API-Key`` header (v2 rejects keys in the query string).
    Every sub-fetch is optional: a failing detail endpoint only moves the field
    into ``unavailable_fields``.
    """

    def __init__(self, api_key: str | None = None, client_factory=None):
        self._api_key = (api_key or os.getenv("ONET_API_KEY", "")).strip()
        self._client_factory = client_factory or httpx.Client

    def _get(self, http, path: str, params: dict | None = None) -> dict:
        response = http.get(
            f"{ONET_BASE_URL}{path}",
            params=params,
            headers={"Accept": "application/json", "X-API-Key": self._api_key},
        )
        response.raise_for_status()
        return response.json()

    def _elements(self, http, code: str, endpoint: str, cap: int,
                  sort_by: str | None = None) -> list[dict]:
        payload = self._get(http, f"/online/occupations/{code}/{endpoint}")
        items = list(payload.get("element") or [])
        if sort_by:
            # Document order is O*NET's content-model order (generic basic skills
            # first); rank by the requested metric so the cap keeps the occupation's
            # most important entries rather than the first N listed.
            items.sort(key=lambda item: _scale(item.get(sort_by)) or 0.0, reverse=True)
        return items[:cap]

    def benchmark(self, role: str) -> dict:
        if not self._api_key:
            return EmptyBenchmark().benchmark(role)
        role = str(role or "").strip()
        benchmark = {
            "role": role,
            "provider": "O*NET Web Services v2",
            "source": SOURCE,
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "title": role,
            "onet_code": None,
            "description": "",
            "core_skills": [],
            "tasks": [],
            "education": None,
            "job_zone": None,
            "related_titles": [],
            "outlook": None,
            "unavailable_fields": [],
        }
        unavailable = benchmark["unavailable_fields"]
        try:
            with self._client_factory(timeout=HTTP_TIMEOUT) as http:
                found = (self._get(http, "/online/search", {"keyword": role, "end_clip": "5"})).get("occupation") or []
                if not found:
                    benchmark["unavailable_fields"].append("all (no O*NET occupation matched)")
                    benchmark["benchmark_quality"] = 0.0
                    return benchmark
                top = found[0]
                code, title = str(top.get("code", "")), str(top.get("title") or role)
                benchmark["onet_code"], benchmark["title"] = code, title
                benchmark["related_titles"] = [
                    str(o.get("title")) for o in found[1:] if o.get("title")
                ][:4]
                try:
                    items = self._elements(http, code, "details/skills", 10, sort_by="importance")
                    benchmark["core_skills"] = [
                        {
                            "name": str(item.get("name")),
                            "importance": _scale(item.get("importance")),
                            "level": _scale(item.get("level")),
                            "required_level": importance_to_level(_scale(item.get("importance"))),
                        }
                        for item in items if item.get("name")
                    ]
                    if not benchmark["core_skills"]:
                        unavailable.append("core_skills")
                except Exception:
                    unavailable.append("core_skills")
                try:
                    benchmark["tasks"] = [
                        str(item.get("name"))[:200] for item in self._elements(http, code, "details/tasks", 6)
                        if item.get("name")
                    ]
                    if not benchmark["tasks"]:
                        unavailable.append("tasks")
                except Exception:
                    unavailable.append("tasks")
                try:
                    knowledge = [
                        str(item.get("name")) for item in self._elements(http, code, "summary/knowledge", 5)
                        if item.get("name")
                    ]
                    if knowledge:
                        benchmark["knowledge_areas"] = knowledge
                except Exception:
                    pass  # knowledge is optional enrichment
                try:
                    education = self._get(http, f"/online/occupations/{code}/summary/education")
                    entries = education.get("education_level") or []
                    usually = next((e for e in entries if e.get("score") == 70), None)
                    if usually and usually.get("name"):
                        benchmark["education"] = str(usually["name"])
                    else:
                        unavailable.append("education")
                except Exception:
                    unavailable.append("education")
                try:
                    overview = self._get(http, f"/online/occupations/{code}/")
                    benchmark["job_zone"] = overview.get("job_zone")
                    tags = overview.get("tags") or {}
                    benchmark["bright_outlook"] = bool(tags.get("bright_outlook"))
                    benchmark["description"] = str(overview.get("description") or "").strip()[:400]
                except Exception:
                    unavailable.extend(["job_zone", "outlook", "description"])
                try:
                    outlook = self._get(http, f"/online/occupations/{code}/summary/outlook")
                    growth = outlook.get("growth_rate") or {}
                    benchmark["outlook"] = {
                        "category": str(outlook.get("outlook") or ""),
                        "growth": str(growth.get("category") or ""),
                    }
                except Exception:
                    if "outlook" not in unavailable:
                        unavailable.append("outlook")
        except Exception:
            # search itself failed: keep only what we truly know (the role name)
            return {
                **benchmark,
                "title": role,
                "onet_code": None,
                "unavailable_fields": ["all (O*NET request failed)"],
                "benchmark_quality": 0.0,
            }
        benchmark["benchmark_quality"] = round(
            1.0 - 0.12 * len([f for f in unavailable if not f.startswith("all")]), 2
        )
        return benchmark


def facts_from_benchmark(benchmark: dict) -> dict:
    """Flat, card-ready facts describing the benchmark itself (data about the
    target role — explicitly separate from anything about the user)."""
    facts: dict = {}
    if benchmark.get("onet_code"):
        facts["O*NET code"] = benchmark["onet_code"]
    related = "; ".join(benchmark.get("related_titles") or [])
    if related:
        facts["Related titles"] = related
    skills = benchmark.get("core_skills") or []
    if skills:
        facts["Typical skills"] = "; ".join(s["name"] for s in skills[:6])
    if benchmark.get("description"):
        facts["What they do"] = benchmark["description"][:300]
    outlook = benchmark.get("outlook") or {}
    if outlook.get("category") or benchmark.get("bright_outlook"):
        label = outlook.get("category") or "Bright Outlook"
        facts["Outlook"] = f"{label} ({outlook['growth']})" if outlook.get("growth") else str(label)
    if benchmark.get("job_zone") is not None:
        facts["Job zone"] = str(benchmark["job_zone"])
    if benchmark.get("education"):
        facts["Typical education"] = benchmark["education"]
    if benchmark.get("unavailable_fields"):
        facts["Benchmark gaps"] = "; ".join(benchmark["unavailable_fields"][:6])
    if benchmark.get("retrieved_at"):
        facts["Benchmark retrieved"] = benchmark["retrieved_at"][:10]
    facts["Benchmark source"] = benchmark.get("source") or "none (role name only)"
    return facts


def provider() -> BenchmarkProvider:
    """Provider selection point: O*NET when a key is configured, else null-object."""
    return OnetProvider() if os.getenv("ONET_API_KEY", "").strip() else EmptyBenchmark()


def dumps(benchmark: dict) -> str:
    return json.dumps(benchmark)
