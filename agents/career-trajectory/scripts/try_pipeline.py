"""Try the full PathFinder pipeline locally: real résumé + mock LinkedIn history + target role.

Default mode drives the deterministic engine directly — no API keys, no network
(except a live O*NET benchmark when ONET_API_KEY is set; it degrades gracefully
without one). ``--chat`` additionally runs the real ASI:One agent loop, the same
orchestration ASI:One would trigger.

Usage:
  # pipeline only (deterministic, key-free):
  uv run python scripts/try_pipeline.py --resume my_cv.pdf --history scripts/mock_linkedin.example.json --role "Data Scientist"

  # same, plus the real LLM loop (needs ASI_ONE_API_KEY):
  uv run python scripts/try_pipeline.py --resume my_cv.pdf --history scripts/mock_linkedin.example.json --role "Data Scientist" --chat

Outputs the unified profile, conflicts, scores, gaps, path options, and writes
the validated Goal Planning payload to --out (default: /tmp/trajectory_payload.json).
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import career_profile as pf
import onet
import resume as resume_mod
import scoring
import trajectories


def load_history(path: str, as_of: date) -> dict:
    result = pf.normalize_input(Path(path).read_text(), provenance=pf.PROVENANCE_LINKEDIN, as_of=as_of)
    profile = result["profile"]
    print(f"\n[linkedin history] roles={len(profile['experiences'])} "
          f"skills={len(profile['skills'])} warnings={len(result['warnings'])}")
    for warning in result["warnings"]:
        print(f"  ! {warning}")
    return profile


def load_resume(path: str, as_of: date) -> tuple[dict, list]:
    parsed = resume_mod.parse_resume_bytes(path, Path(path).read_bytes(), as_of=as_of)
    if parsed.get("status") != "ok":
        print(f"\n[résumé] could not parse: {parsed.get('error')}")
        raise SystemExit(1)
    fragment, warnings = parsed["profile"], parsed.get("warnings") or []
    print(f"\n[résumé {Path(path).suffix or '?'}] roles={len(fragment['experiences'])} "
          f"skills={len(fragment['skills'])} projects={len(fragment['projects'])} "
          f"certs={len(fragment['certifications'])} warnings={len(warnings)}")
    for warning in warnings:
        print(f"  ! {warning}")
    return fragment, warnings


def build_profile(args, as_of: date) -> tuple[dict, dict, str]:
    profile = load_history(args.history, as_of) if args.history else None
    fragment: dict = {}
    resume_json = ""
    if args.resume:
        fragment, _ = load_resume(args.resume, as_of)
        resume_json = json.dumps(fragment)
    if profile and fragment:
        profile = pf.reconcile(profile, fragment)
    else:
        profile = profile or fragment
    if not profile or not (profile.get("experiences") or profile.get("skills")):
        print("\nNo career evidence found - check the inputs.")
        raise SystemExit(1)
    return profile, fragment, resume_json


def show(profile: dict, analysis: dict, plan: dict, benchmark: dict) -> None:
    print("\n=== Unified profile ===")
    print(f"  profile_id: {profile['profile_id']}  completeness: {analysis['completeness']:.0%}")
    for role in profile["experiences"]:
        print(f"  - {role.get('title') or '(untitled)'} @ {role.get('company') or '?'} "
              f"{role.get('start') or '?'} -> {role.get('end') or '?'}  provenance={role.get('provenance')}")
    for conflict in profile.get("conflicts") or []:
        print(f"  CONFLICT [{conflict['kind']}] {conflict['summary']}  <- needs your confirmation")
    print("\n=== Scores (guidance, not ranking) ===")
    print(f"  readiness {analysis['readiness_score']}/100 | transferability "
          f"{analysis['transferability_score']}/100 | evidence {analysis['evidence_strength_score']}/100 "
          f"| difficulty {analysis['transition_difficulty']} | confidence {analysis['confidence']}")
    benchmark_note = benchmark.get("provider") == "none" or bool(benchmark.get("unavailable_fields"))
    if benchmark_note:
        print(f"  benchmark: {benchmark.get('provider')} "
              f"(unavailable: {', '.join(benchmark.get('unavailable_fields') or ['none'])})")
    print("\n=== Gaps ===")
    for gap in analysis["gaps"][:8]:
        print(f"  [{gap['priority']:6}] {gap['skill']}: level {gap['current_level']} -> "
              f"{gap['required_level']} ({gap['gap_type']})")
    if not analysis["gaps"]:
        print("  none above threshold")
    print("\n=== Path options ===")
    for row in plan["comparison"]:
        marker = "  <-- recommended" if row["id"] == plan["recommended_id"] else ""
        print(f"  {row['id']}: {row['title']} | ~{row['estimated_months']} months | "
              f"gaps {row['high_priority_gaps']} | constraint fit {row['constraint_fit']} | "
              f"confidence {row['confidence']:.0%}{marker}")


def chat_mode(profile: dict, fragment: dict, resume_json: str, args, as_of: date) -> None:
    import brain  # lazy: requires ASI_ONE_API_KEY

    parts = []
    if args.history:
        parts.append(Path(args.history).read_text().strip())
    if fragment:
        parts.append(f"[A résumé file was received and parsed deterministically. Use analyze_readiness "
                     f"with this stored resume fragment JSON from context: {resume_json[:2000]}]")
    parts.append(f"Analyze this career evidence and plan trajectory options toward {args.role or 'the target role'}.")
    context = json.dumps({"profile": pf.summarize_for_prompt(profile), "target_role": args.role})[:4000]
    prompt = "\n\n".join(parts) + f"\n\n[Session context: {context}]"
    reply, data = asyncio.run(brain.answer(prompt))
    print("\n=== Agent reply (real ASI:One loop) ===")
    print(reply[:1500])
    if data and data.get("paths"):
        print("\n[tool data] paths:", [p["id"] for p in data["paths"]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", help="résumé file: .pdf/.docx/.txt/.json")
    parser.add_argument("--history", help="mock LinkedIn alumni_history JSON file")
    parser.add_argument("--role", default="", help="target role, e.g. 'Data Scientist'")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="evaluation date (YYYY-MM-DD)")
    parser.add_argument("--out", default="/tmp/trajectory_payload.json", help="where to write the payload")
    parser.add_argument("--chat", action="store_true", help="also run the real ASI:One agent loop")
    args = parser.parse_args()
    if not args.resume and not args.history:
        parser.error("provide --resume and/or --history")

    as_of = date.fromisoformat(args.as_of)
    profile, fragment, resume_json = build_profile(args, as_of)

    role = args.role or (profile.get("target_roles") or [""])[0]
    if not role:
        parser.error("no target role found: pass --role")
    if role not in (profile.get("target_roles") or []):
        profile["target_roles"] = (profile.get("target_roles") or []) + [role]

    benchmark = onet.provider().benchmark(role)
    analysis = scoring.analyze(profile, benchmark, as_of=as_of)
    plan = trajectories.plan(profile, analysis, benchmark)

    show(profile, analysis, plan, benchmark)

    selection = trajectories.select(plan, analysis, plan["recommended_id"])
    payload = trajectories.build_payload(profile, analysis, selection)
    errors = trajectories.validate_payload(payload)
    Path(args.out).write_text(trajectories.dumps(payload))
    print("\n=== Goal Planning payload ===")
    print(f"  {len(errors) if errors else '0'} validation errors -> written to {args.out}")
    print(f"  trajectory_id={payload['trajectory_id']} waypoints={len(payload['waypoints'])} "
          f"alternatives={len(payload['alternatives'])}")
    for error in errors:
        print(f"  ! {error}")

    if args.chat:
        chat_mode(profile, fragment, resume_json, args, as_of)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
