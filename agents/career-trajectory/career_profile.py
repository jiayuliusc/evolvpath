"""Canonical career-profile schema (v2.0), evidence classification, and reconciliation.

This module owns the deterministic "career evidence" layer of PathFinder:

- ``normalize_input``: accepts the legacy ``alumni_history`` JSON (v1), the
  versioned canonical input (v2), a bare list of roles, or pasted free text,
  and returns one canonical profile dict plus warnings.
- Evidence classification: every skill carries evidence items with a level
  (``demonstrated`` / ``applied`` / ``mentioned`` / ``inferred``), a provenance
  source (``resume`` / ``linkedin`` / ``user`` / ``onet``) and a quantified flag.
  A skill's proficiency level (0-4) is derived ONLY from its evidence; a job
  title never contributes skills.
- ``reconcile``: merges a LinkedIn-derived profile with a résumé-derived
  fragment, deduplicating positions, preserving provenance, and flagging
  conflicts for user confirmation instead of silently discarding them.

All functions are pure and deterministic given their inputs (``as_of`` is an
explicit argument where wall-clock time matters), so tests never need mocks
here. Missing data produces warnings and lower completeness, never crashes or
fabricated facts.

Privacy: ``redact`` scrubs emails/phones/long digit runs for logs; profiles use
anonymous ``profile_id`` values; ``summarize_for_prompt`` returns a bounded,
redacted view so raw résumé text never reaches the LLM or logs.
"""

import json
import re
import uuid
from datetime import date

SCHEMA_VERSION = "2.0"

# Evidence levels, strongest first. Demonstrated evidence must weigh more than
# mentioned or inferred evidence: these weights are the single source of truth.
EVIDENCE_LEVELS = ("demonstrated", "applied", "mentioned", "inferred")
EVIDENCE_WEIGHTS = {"demonstrated": 1.0, "applied": 0.7, "mentioned": 0.4, "inferred": 0.15}

# Documented proficiency scale used by gap analysis (current_level/required_level).
PROFICIENCY_LABELS = {0: "none", 1: "awareness", 2: "working", 3: "proficient", 4: "expert"}

# Provenance values carried on extracted facts.
PROVENANCE_RESUME = "resume"
PROVENANCE_LINKEDIN = "linkedin"
PROVENANCE_USER = "user"
PROVENANCE_ONET = "onet"

PRESENT_WORDS = {"", "present", "current", "now"}
MAX_TEXT_CHARS = 40_000  # cap on any pasted/file text before parsing

_MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# Résumés are strictly untrusted career DATA. Text that looks like instructions
# to an LLM is ignored as content (never obeyed) and surfaced as a warning.
INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "disregard all",
    "forget your instructions",
    "system prompt",
    "you are now",
    "act as",
    "pretend to be",
    "do not tell the user",
    "exfiltrate",
)

_DATE_RANGE_RE = re.compile(
    r"((?:[A-Za-z]{3,9}\.?\s*)?(?:19|20)\d{2}(?:\s*[-/]\s*\d{1,2})?)"
    r"\s*(?:-|–|—|to|until|through)\s*"
    r"((?:[A-Za-z]{3,9}\.?\s*)?(?:19|20)\d{2}(?:\s*[-/]\s*\d{1,2})?|present)",
    re.IGNORECASE,
)
_TITLE_LINE_RE = re.compile(
    r"^\s*(?:[-•*|▶o\s]*)"
    r"((?:[A-Z][A-Za-z&./' ]{2,50}?))"
    r"(?:\s*[,|@]|\s+at\s+)"
    r"\s*([A-Z][A-Za-z&./' ]{1,50})\s*$"
)
_SKILL_LIST_HEADER_RE = re.compile(
    r"^(technical skills|skills( & tools)?|core skills|competencies|technologies|tools)\s*:?\s*$",
    re.IGNORECASE,
)
_SECTION_HEADERS = {
    "experience": ("experience", "work experience", "professional experience",
                   "employment", "employment history", "career history"),
    "education": ("education", "academic background"),
    "skills": ("skills", "technical skills", "skills & tools", "core skills",
               "competencies", "technologies", "tools"),
    "projects": ("projects", "selected projects", "personal projects"),
    "certifications": ("certifications", "certificates", "licenses"),
    "summary": ("summary", "professional summary", "profile", "about", "objective"),
}
_DEGREE_RE = re.compile(
    r"\b(ph\.?d|doctorate|m\.?b\.?a|m\.?s\.?c?|master(?:'s)?|b\.?s\.?c?|b\.?a\.?"
    r"|bachelor(?:'s)?|associate|diploma)\b", re.IGNORECASE,
)
_QUANTIFIED_RE = re.compile(r"(\d+(?:[.,]\d+)?\s*%|\$\s?\d|\b\d+(?:[.,]\d+)?\s?(?:k\b|x\b|users|customers|clients|people|engineers|reports|records|projects|hours|weeks|months|years))", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-•*▪·]|o\s|\d+[.)]\s)")
_PII_RE = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[redacted-email]"),
    (re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)"), "[redacted-number]"),
    (re.compile(r"\bhttps?://\S+"), "[redacted-url]"),
]


# --- dates and durations -----------------------------------------------------


def parse_month(value, as_of: date | None = None) -> tuple[int, int] | None:
    """Parse a date-ish value into (year, month).

    Accepts ``YYYY-MM``/``YYYY/MM``, ``YYYY``, ``Mon YYYY``/``Month YYYY``,
    ``YYYY-MM-DD`` and present-tense words. ``present`` resolves to ``as_of``
    (evaluation date) so tenure is reproducible for identical inputs.
    """
    if value is None:
        return None
    text = str(value).strip().lower().rstrip(".")
    if text in PRESENT_WORDS:
        if as_of is None:
            return None
        return (as_of.year, as_of.month)
    if not text:
        return None
    named = re.match(r"^([a-z]{3,9})\.?\s+((?:19|20)\d{2})$", text)
    if named and named.group(1) in _MONTH_NAMES:
        return (int(named.group(2)), _MONTH_NAMES[named.group(1)])
    match = re.match(r"^(?:[a-z]{3,9}\.?\s*)?((?:19|20)\d{2})(?:[-/](\d{1,2}))?(?:-\d{1,2})?$", text)
    if match:
        year = int(match.group(1))
        month = min(max(int(match.group(2) or 1), 1), 12)
        return (year, month)
    return None


def is_present(value) -> bool:
    return str(value or "").strip().lower() in PRESENT_WORDS


def duration_years(start, end, as_of: date | None = None) -> float | None:
    """Tenure in years between two date-ish values; None when either is unparseable."""
    begin, finish = parse_month(start, as_of), parse_month(end, as_of)
    if begin is None or finish is None:
        return None
    months = (finish[0] - begin[0]) * 12 + (finish[1] - begin[1])
    return max(months, 0) / 12


# --- normalization helpers ----------------------------------------------------


def redact(text) -> str:
    """Scrub emails, phone-like runs and URLs for logs and error messages."""
    text = str(text or "")
    for pattern, replacement in _PII_RE:
        text = pattern.sub(replacement, text)
    return text[:400]


def normalize_skill(name) -> str:
    """Matching key for a skill: lowercase, collapsed punctuation, singular-ish."""
    key = re.sub(r"[^a-z0-9+#. ]", " ", str(name or "").lower())
    key = re.sub(r"\s+", " ", key).strip()
    return key.rstrip("s") if len(key) > 4 and key.endswith("s") and not key.endswith("ss") else key


def same_company(a, b) -> bool:
    ka, kb = normalize_skill(a), normalize_skill(b)  # same normalization rules as skills
    if not ka or not kb:
        return False
    return ka == kb or (len(ka) >= 4 and ka in kb) or (len(kb) >= 4 and kb in ka)


def new_profile_id() -> str:
    """Anonymous, non-derived-from-PII identifier for downstream payloads."""
    return f"anon-{uuid.uuid4().hex[:8]}"


def _evidence(level: str, source: str, detail: str = "", quantified: bool = False) -> dict:
    return {"level": level, "source": source, "detail": redact(detail)[:200], "quantified": bool(quantified)}


def level_from_evidence(evidence: list) -> int:
    """Proficiency 0-4 derived ONLY from evidence items (never from job titles).

    quantified demonstrated > demonstrated > applied > mentioned/inferred.
    """
    best = 0
    for item in evidence or []:
        tier = item.get("level") if item.get("level") in EVIDENCE_WEIGHTS else "mentioned"
        base = {"demonstrated": 3, "applied": 2, "mentioned": 1, "inferred": 1}[tier]
        score = base + (1 if tier == "demonstrated" and item.get("quantified") else 0)
        best = max(best, min(score, 4))
    return best


def top_evidence_tier(evidence: list) -> str:
    for tier in EVIDENCE_LEVELS:
        if any((item.get("level") if item.get("level") in EVIDENCE_WEIGHTS else "mentioned") == tier
               for item in evidence or []):
            return tier
    return "inferred"


def _add_skill(profile: dict, name: str, item: dict) -> None:
    name = str(name or "").strip()
    if not name or len(name) > 60:
        return
    key = normalize_skill(name)
    if not key:
        return
    for skill in profile["skills"]:
        if skill["key"] == key:
            skill["evidence"].append(item)
            skill["level"] = level_from_evidence(skill["evidence"])
            skill["tier"] = top_evidence_tier(skill["evidence"])
            return
    profile["skills"].append({
        "name": name,
        "key": key,
        "level": level_from_evidence([item]),
        "tier": top_evidence_tier([item]),
        "evidence": [item],
    })


# --- input normalization -------------------------------------------------------


def empty_profile(profile_id: str | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id or new_profile_id(),
        "name": "",
        "headline": "",
        "location": "",
        "skills": [],
        "education": [],
        "experiences": [],
        "projects": [],
        "certifications": [],
        "target_roles": [],
        "preferences": {},
        "constraints": [],
        "source_metadata": [],
        "warnings": [],
        "conflicts": [],
    }


def normalize_input(raw, provenance: str = PROVENANCE_USER, as_of: date | None = None) -> dict:
    """Normalize any accepted career input into ``{"profile": ..., "warnings": [...]}``.

    Accepts the legacy ``alumni_history`` JSON object (v1 aliases included), the
    versioned canonical v2 object, a bare JSON array of roles, or pasted free
    text. Raises ``ValueError`` only when no career evidence at all is present.
    """
    warnings: list[str] = []
    profile = empty_profile()
    data, text = None, (raw if isinstance(raw, str) else "").strip()
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, list):
        data = {"experiences": [r for r in raw if isinstance(r, dict)]}
    elif text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
            data = {"experiences": parsed} if isinstance(parsed, list) else parsed
        except ValueError:
            data = None
    if isinstance(data, dict) and data:
        _from_structured(data, profile, warnings, provenance)
    else:
        _from_text(text, profile, warnings, provenance)
    if as_of is not None:
        profile["as_of"] = as_of.isoformat()
    profile["warnings"] = warnings
    return {"profile": profile, "warnings": warnings}


def _clean_role(entry: dict, provenance: str, warnings: list) -> dict | None:
    title = str(entry.get("title") or entry.get("role") or "").strip()
    company = str(
        entry.get("company") or entry.get("organisation") or entry.get("organization") or entry.get("employer") or ""
    ).strip()
    start = str(entry.get("start") or entry.get("start_date") or "").strip()
    end = str(entry.get("end") or entry.get("end_date") or "").strip()
    if not title and not company and not start:
        return None
    role = {
        "title": title,
        "company": company,
        "start": start,
        "end": end,
        "skills": [str(s).strip() for s in (entry.get("skills") or []) if str(s).strip()][:20],
        "achievements": [],
        "provenance": [provenance],
    }
    if start and parse_month(start) is None:
        warnings.append(f"Unparseable start date '{redact(start)}'; role kept but tenure ignored.")
    if end and not is_present(end) and parse_month(end) is None:
        warnings.append(f"Unparseable end date '{redact(end)}'; role kept but tenure ignored.")
    if not start:
        warnings.append(f"Role '{title or company or 'unnamed'}' has no start date; treated as undated.")
    for ach in entry.get("achievements") or entry.get("outcomes") or []:
        text = str(ach).strip()
        if text:
            role["achievements"].append({"text": redact(text)[:300], "quantified": bool(_QUANTIFIED_RE.search(text))})
    return role


def _from_structured(data: dict, profile: dict, warnings: list, provenance: str) -> None:
    if not data.get("schema_version"):
        warnings.append(
            "No schema_version in structured input; read as legacy alumni_history (canonical v2 compatible)."
        )
    profile["name"] = str(data.get("name") or "").strip()
    profile["headline"] = str(data.get("headline") or "").strip()
    profile["location"] = str(data.get("location") or "").strip()
    if data.get("profile_id"):
        profile["profile_id"] = str(data["profile_id"])[:64]
    targets = data.get("target_roles") or ([data["target_role"]] if data.get("target_role") else [])
    profile["target_roles"] = [str(t).strip() for t in targets if str(t).strip()][:5]
    prefs = data.get("preferences")
    profile["preferences"] = prefs if isinstance(prefs, dict) else {}
    constraints = data.get("constraints")
    profile["constraints"] = [str(c) for c in constraints if str(c).strip()][:10] if isinstance(constraints, list) else []
    meta = data.get("source_metadata")
    profile["source_metadata"] = meta if isinstance(meta, list) else []

    raw_skills = data.get("skills") or []
    if isinstance(raw_skills, str):  # legacy: skills as one comma-separated string
        raw_skills = raw_skills.split(",")
    for skill in raw_skills:
        if isinstance(skill, str):
            for part in skill.split(","):
                _add_skill(profile, part.strip(), _evidence("mentioned", provenance))
        elif isinstance(skill, dict):
            _add_skill(profile, skill.get("name"), _evidence(
                str(skill.get("level")) if str(skill.get("level")) in EVIDENCE_WEIGHTS else "mentioned",
                str(skill.get("source") or provenance), str(skill.get("detail") or "")))

    roles = data.get("experiences") or data.get("roles") or data.get("positions") or []
    for entry in roles if isinstance(roles, list) else []:
        if not isinstance(entry, dict):
            continue
        role = _clean_role(entry, provenance, warnings)
        if role is None:
            continue
        profile["experiences"].append(role)
        for skill_name in role["skills"]:
            _add_skill(profile, skill_name, _evidence("applied", provenance, role["title"]))
        for achievement in role["achievements"]:
            for skill_name in _skills_in_text(achievement["text"], [s["name"] for s in profile["skills"]]):
                _add_skill(profile, skill_name, _evidence("demonstrated", provenance, role["title"],
                                                          achievement["quantified"]))

    for entry in data.get("projects") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("title") or "").strip()
        description = str(entry.get("description") or entry.get("summary") or "").strip()
        skills = [str(s).strip() for s in (entry.get("skills") or []) if str(s).strip()]
        if not name and not description:
            continue
        project = {"name": redact(name)[:120], "description": redact(description)[:400],
                   "skills": skills[:15], "provenance": [provenance]}
        profile["projects"].append(project)
        for skill_name in skills or _skills_in_text(description, []):
            _add_skill(profile, skill_name, _evidence("demonstrated", provenance, project["name"]))

    for entry in data.get("certifications") or []:
        if isinstance(entry, str) and entry.strip():
            entry = {"name": entry}
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            continue
        profile["certifications"].append({
            "name": redact(str(entry.get("name")))[:120],
            "issuer": redact(str(entry.get("issuer") or ""))[:80],
            "year": str(entry.get("year") or "")[:10],
            "provenance": [provenance],
        })

    for entry in data.get("education") or []:
        if not isinstance(entry, dict):
            if isinstance(entry, str) and entry.strip():
                entry = {"degree": entry}
            else:
                continue
        degree = str(entry.get("degree") or entry.get("level") or "").strip()
        school = str(entry.get("school") or entry.get("institution") or "").strip()
        if not degree and not school:
            continue
        profile["education"].append({
            "degree": redact(degree)[:80], "field": redact(str(entry.get("field") or ""))[:80],
            "school": redact(school)[:80], "end": str(entry.get("end") or entry.get("year") or "")[:10],
            "provenance": [provenance],
        })

    if not profile["experiences"] and not profile["skills"] and not profile["projects"]:
        raise ValueError("no experiences, skills or projects found in structured input")


# --- free-text / résumé parsing ------------------------------------------------


def _skills_in_text(text: str, known: list[str]) -> list[str]:
    """Skills explicitly named inside an achievement/project sentence.

    Only exact word matches against the profile's own skill names count, so a
    quantified outcome upgrades a skill the user actually listed — it never
    invents a skill from a job title.
    """
    lowered = f" {str(text).lower()} "
    found = []
    for name in known:
        needle = f" {normalize_skill(name)}"
        if needle in lowered and name not in found:
            found.append(name)
    return found[:10]


def _split_sections(text: str) -> dict[str, list[str]]:
    """Split raw text into rough résumé sections via header lines."""
    sections: dict[str, list[str]] = {"__top__": []}
    current = "__top__"
    for line in text.splitlines()[:400]:
        stripped = line.strip().rstrip(":").lower()
        matched = None
        for section, headers in _SECTION_HEADERS.items():
            if stripped in headers or (stripped.isupper() and stripped in
                                       {h.upper() for h in headers}):
                matched = section
                break
        if matched:
            current = matched
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line[:300])
    return sections


def _parse_experience_lines(lines: list[str], provenance: str, warnings: list) -> list[dict]:
    """Pair 'Title, Company' lines with date ranges found nearby (order-based)."""
    text = "\n".join(lines)
    ranges = _DATE_RANGE_RE.findall(text)
    roles: list[dict] = []
    pending_title = ""
    for line in lines:
        match = _TITLE_LINE_RE.match(line)
        if match:
            pending_title = f"{match.group(1).strip(' .')}|{match.group(2).strip(' .')}"
            continue
        for start, end in _DATE_RANGE_RE.findall(line):
            title, company = (pending_title.split("|", 1) + [""])[:2] if pending_title else ("", "")
            pending_title = ""
            role = {"title": title, "company": company, "start": start.strip(), "end": end.strip(),
                    "skills": [], "achievements": [], "provenance": [provenance]}
            if not start or parse_month(start) is None or (not is_present(end) and parse_month(end) is None):
                warnings.append("Ambiguous dates around a role; tenure for it is treated as unknown.")
            roles.append(role)
    if not roles and ranges:
        for start, end in ranges:
            roles.append({"title": "", "company": "", "start": start.strip(), "end": end.strip(),
                          "skills": [], "achievements": [], "provenance": [provenance]})
    return roles


def _parse_resume_text(text: str, provenance: str, warnings: list) -> dict:
    """Section-aware extraction of career facts from a résumé/text blob."""
    fragment = empty_profile()
    sections = _split_sections(text)
    experience_lines = sections.get("experience") or []
    if not experience_lines:  # short pasted histories have no headers at all
        experience_lines = text.splitlines()
    roles = _parse_experience_lines(experience_lines, provenance, warnings)
    fragment["experiences"] = roles

    skill_lines = sections.get("skills") or []
    inline_skill_lines = [line for line in text.splitlines()
                          if re.match(r"^\s*(technical )?skills\s*:\s*\S", line, re.IGNORECASE)]
    skill_lines = skill_lines + inline_skill_lines
    if skill_lines:
        blob = " ".join(skill_lines)
        for name in re.split(r"[,;•|/]|\s{3,}", blob):
            _add_skill(fragment, name, _evidence("mentioned", provenance))

    summary = " ".join(sections.get("summary") or [])[:400]
    if summary:
        for name in _skills_in_text(summary, [n for n in re.split(r"[,;/]\s*", summary) if 2 < len(n) < 40]):
            _add_skill(fragment, name, _evidence("mentioned", provenance))

    for section_name, target in (("projects", fragment["projects"]), ("certifications", fragment["certifications"])):
        for line in sections.get(section_name) or []:
            clean = _BULLET_RE.sub("", line).strip(" .")
            if len(clean) < 4:
                continue
            entry = {"name": redact(clean.split(",")[0])[:120], "description": redact(clean)[:300],
                     "skills": [], "provenance": [provenance]} if section_name == "projects" else {
                "name": redact(clean)[:120], "issuer": "", "year": "", "provenance": [provenance]}
            target.append(entry)

    for line in sections.get("education") or []:
        if _DEGREE_RE.search(line):
            clean = _BULLET_RE.sub("", line).strip(" .")
            years = re.search(r"((?:19|20)\d{2})", line)
            fragment["education"].append({
                "degree": redact(clean)[:120], "field": "", "school": "",
                "end": years.group(1) if years else "", "provenance": [provenance],
            })

    for role in roles:
        context = [r for r in experience_lines]
        idx = None
        for i, line in enumerate(context):
            if role["start"] and role["start"] in line:
                idx = i
                break
        window = context[idx: idx + 12] if idx is not None else []
        for raw_line in window:
            if not _BULLET_RE.match(raw_line) and "skill" not in raw_line.lower():
                continue
            clean = _BULLET_RE.sub("", raw_line).strip()
            if _QUANTIFIED_RE.search(clean):
                role["achievements"].append({"text": redact(clean)[:300], "quantified": True})
            elif re.match(r"^skills?\s*:", clean, re.IGNORECASE):
                role["skills"] = [p.strip() for p in re.split(r"[,;]", clean.split(":", 1)[1]) if p.strip()][:15]
    for role in roles:
        for achievement in role["achievements"]:
            for name in _skills_in_text(achievement["text"], [s["name"] for s in fragment["skills"]]):
                _add_skill(fragment, name, _evidence("demonstrated", provenance, role["title"], True))
        for skill_name in role["skills"]:
            _add_skill(fragment, skill_name, _evidence("applied", provenance, role["title"]))
    return fragment


def _from_text(text: str, profile: dict, warnings: list, provenance: str) -> None:
    if not text.strip():
        raise ValueError("empty input")
    if len(text) > MAX_TEXT_CHARS:
        warnings.append(f"Input truncated to {MAX_TEXT_CHARS} characters before parsing.")
        text = text[:MAX_TEXT_CHARS]
    lowered = text.lower()
    injection_hits = [p for p in INJECTION_PATTERNS if p in lowered]
    if injection_hits:
        warnings.append(
            "Input contained instruction-like text; it was ignored as career data and never acted on."
        )
    fragment = _parse_resume_text(text, provenance, warnings)
    headline = fragment["experiences"] and fragment["experiences"][0]["title"] or ""
    fragment["headline"] = headline
    if not fragment["experiences"] and not fragment["skills"] and not fragment["projects"]:
        raise ValueError("no roles, date ranges, skills or projects found")
    for warning in fragment["warnings"]:
        if warning not in warnings:
            warnings.append(warning)
    _merge_into(profile, fragment)


def _merge_into(base: dict, fragment: dict) -> None:
    """Fold a parsed fragment into a profile, deduping skills by key."""
    for skill in fragment["skills"]:
        for item in skill["evidence"]:
            _add_skill(base, skill["name"], item)
    base["experiences"].extend(fragment["experiences"])
    base["education"].extend(fragment["education"])
    base["projects"].extend(fragment["projects"])
    base["certifications"].extend(fragment["certifications"])
    base["headline"] = base["headline"] or fragment["headline"]


# --- reconciliation -------------------------------------------------------------


def _month_index(pair: tuple | None) -> float | None:
    return pair[0] * 12 + pair[1] if pair else None


def _overlap(a: tuple, b: tuple) -> bool:
    """Do two ((y, m)|None, (y, m)|None) ranges overlap, with a 3-month grace gap?
    A None bound means open-ended (or unknown), which never closes the range."""
    a_start, a_end = _month_index(a[0]), _month_index(a[1])
    b_start, b_end = _month_index(b[0]), _month_index(b[1])
    return (a_start is None or b_end is None or a_start <= b_end + 3) and \
           (b_start is None or a_end is None or b_start <= a_end + 3)


def reconcile(history_profile: dict, resume_profile: dict) -> dict:
    """Merge a LinkedIn-derived profile with a résumé-derived fragment.

    - Positions at the same company with overlapping dates are deduplicated.
    - Provenance is preserved on every fact (union of both sources).
    - Conflicting titles/dates are flagged in ``conflicts`` with both values and
      ``needs_confirmation: true`` — never silently discarded. The résumé value
      wins as the working copy because it is the user-curated document; the
      flagged conflict lets the user override.
    - Skills are merged by evidence; levels recompute from combined evidence.
    """
    warnings = list(dict.fromkeys(
        list(history_profile.get("warnings") or []) + list(resume_profile.get("warnings") or [])
    ))
    merged = empty_profile(history_profile.get("profile_id") or resume_profile.get("profile_id"))
    merged["name"] = history_profile.get("name") or resume_profile.get("name")
    merged["headline"] = history_profile.get("headline") or resume_profile.get("headline")
    merged["location"] = history_profile.get("location") or resume_profile.get("location")
    merged["target_roles"] = list(dict.fromkeys(
        (history_profile.get("target_roles") or []) + (resume_profile.get("target_roles") or [])))[:5]
    merged["preferences"] = {**(resume_profile.get("preferences") or {}),
                             **(history_profile.get("preferences") or {})}
    merged["constraints"] = list(dict.fromkeys(
        (history_profile.get("constraints") or []) + (resume_profile.get("constraints") or [])))[:10]
    merged["source_metadata"] = list(history_profile.get("source_metadata") or []) + \
        list(resume_profile.get("source_metadata") or [])

    conflicts: list[dict] = []
    history_roles = list(history_profile.get("experiences") or [])
    resume_roles = list(resume_profile.get("experiences") or [])
    matched_resume = set()
    for h_role in history_roles:
        h_bounds = (parse_month(h_role.get("start")), parse_month(h_role.get("end")))
        twin = None
        for i, r_role in enumerate(resume_roles):
            if i in matched_resume:
                continue
            if not same_company(h_role.get("company"), r_role.get("company")):
                continue
            if _overlap(h_bounds, (parse_month(r_role.get("start")), parse_month(r_role.get("end")))):
                twin = i
                break
        if twin is None:
            merged_role = {**h_role, "provenance": list(h_role.get("provenance") or [PROVENANCE_LINKEDIN])}
            merged["experiences"].append(merged_role)
            continue
        matched_resume.add(twin)
        r_role = resume_roles[twin]
        role = {
            "title": r_role.get("title") or h_role.get("title"),
            "company": h_role.get("company") or r_role.get("company"),
            "start": r_role.get("start") or h_role.get("start"),
            "end": r_role.get("end") or h_role.get("end"),
            "skills": list(dict.fromkeys((r_role.get("skills") or []) + (h_role.get("skills") or [])))[:20],
            "achievements": list(r_role.get("achievements") or []) + list(h_role.get("achievements") or []),
            "provenance": list(dict.fromkeys(
                (r_role.get("provenance") or []) + (h_role.get("provenance") or []))),
        }
        if r_role.get("title") and h_role.get("title") and \
                normalize_skill(r_role["title"]) != normalize_skill(h_role["title"]):
            conflicts.append({
                "kind": "title_mismatch",
                "summary": f"{role['company']}: LinkedIn says '{h_role['title']}', résumé says '{r_role['title']}'.",
                "linkedin_value": h_role["title"], "resume_value": r_role["title"],
                "needs_confirmation": True,
            })
        if r_role.get("start") and h_role.get("start") and \
                parse_month(r_role["start"]) != parse_month(h_role["start"]):
            conflicts.append({
                "kind": "dates_mismatch",
                "summary": f"{role['company']}: LinkedIn start {h_role['start']}, résumé start {r_role['start']}.",
                "linkedin_value": h_role["start"], "resume_value": r_role["start"],
                "needs_confirmation": True,
            })
        merged["experiences"].append(role)

    for i, r_role in enumerate(resume_roles):
        if i not in matched_resume:
            merged["experiences"].append({**r_role, "provenance": list(r_role.get("provenance") or [PROVENANCE_RESUME])})

    for skill in (history_profile.get("skills") or []) + (resume_profile.get("skills") or []):
        for item in skill.get("evidence") or []:
            _add_skill(merged, skill["name"], item)

    for key in ("education", "certifications", "projects"):
        combined: list[dict] = []
        seen: set[str] = set()
        for entry in (history_profile.get(key) or []) + (resume_profile.get(key) or []):
            dedupe_key = normalize_skill(entry.get("name") or entry.get("degree") or "")
            if dedupe_key and dedupe_key in seen:
                continue
            if dedupe_key:
                seen.add(dedupe_key)
            combined.append(entry)
        merged[key] = combined

    merged["warnings"] = warnings
    merged["conflicts"] = conflicts
    return merged


# --- completeness and LLM-facing summary -----------------------------------------


def completeness(profile: dict) -> float:
    """0-1 estimate of how much career evidence this profile carries.

    Drives confidence and the missing-evidence vs confirmed-gap distinction.
    """
    roles = profile.get("experiences") or []
    dated = [r for r in roles if parse_month(r.get("start")) and
             (is_present(r.get("end")) or parse_month(r.get("end")))]
    skills = profile.get("skills") or []
    evidenced = [s for s in skills if any(
        (item.get("level") in ("demonstrated", "applied")) for item in s.get("evidence") or [])]
    score = 0.0
    score += 0.25 if roles else 0.0
    score += 0.25 * (len(dated) / len(roles) if roles else 0.0)
    score += 0.20 * (len(evidenced) / len(skills) if skills else 0.0)
    score += 0.10 if profile.get("education") else 0.0
    score += 0.10 if any(r.get("achievements") for r in roles) or profile.get("projects") else 0.0
    score += 0.05 if profile.get("certifications") else 0.0
    score += 0.05 if skills else 0.0
    return round(min(score, 1.0), 2)


def summarize_for_prompt(profile: dict) -> dict:
    """Bounded, redacted profile view for the LLM context block.

    Skill *names* and evidence *tiers* pass through; raw résumé paragraphs and
    PII (emails/phones/URLs) never leave this module.
    """
    summary = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile.get("profile_id"),
        "name": bool(profile.get("name")),
        "headline": redact(profile.get("headline") or "")[:80],
        "roles": [
            {"title": r.get("title")[:60], "company": redact(r.get("company"))[:60],
             "start": r.get("start"), "end": r.get("end"),
             "skills": r.get("skills", [])[:10]}
            for r in (profile.get("experiences") or [])[:8]
        ],
        "skills": [
            {"name": s["name"][:40], "tier": s.get("tier"), "level": s.get("level")}
            for s in (profile.get("skills") or [])[:25]
        ],
        "education": [e.get("degree", "")[:60] for e in (profile.get("education") or [])[:4]],
        "certifications": [c.get("name", "")[:60] for c in (profile.get("certifications") or [])[:6]],
        "projects": [p.get("name", "")[:60] for p in (profile.get("projects") or [])[:6]],
        "quantified_achievements": sum(
            1 for r in (profile.get("experiences") or []) for a in r.get("achievements", []) if a.get("quantified")
        ),
        "target_roles": profile.get("target_roles") or [],
        "warnings": (profile.get("warnings") or [])[:8],
        "conflicts": (profile.get("conflicts") or [])[:6],
    }
    return json.loads(json.dumps(summary))  # plain JSON-safe copy


def needs_confirmation(profile: dict) -> list[str]:
    return [c.get("summary", "") for c in (profile.get("conflicts") or []) if c.get("needs_confirmation")]
