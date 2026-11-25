"""Scoring engine, gap analysis, O*NET provider, trajectories, and payload validation.

All occupational-data calls run against the fake httpx client; scoring is pure.
"""

import json

from fixtures import (
    AS_OF,
    DS_BENCHMARK,
    FakeOnetClient,
    career_changer_thin,
    experienced_direct,
    incomplete_dates,
    needs_intermediate,
    onet_default_responses,
    transferable_strong,
    weak_resume_evidence,
)

import career_profile as pf
import onet
import scoring
import trajectories

# --- benchmark provider ----------------------------------------------------------


def _provider():
    return onet.OnetProvider(api_key="test-key", client_factory=FakeOnetClient)


def _use_onet(monkeypatch, responses=None, fail_on=()):
    FakeOnetClient.calls = []
    FakeOnetClient.fail_on = set(fail_on)
    FakeOnetClient.responses = responses if responses is not None else onet_default_responses()
    return _provider()


def test_benchmark_full_success(monkeypatch):
    provider = _use_onet(monkeypatch)
    benchmark = provider.benchmark("Data Scientist")
    assert benchmark["title"] == "Data Scientists"
    assert benchmark["onet_code"] == "15-2051.00"
    assert benchmark["core_skills"][0]["name"] == "Python"
    assert benchmark["core_skills"][0]["required_level"] >= 3
    assert benchmark["job_zone"] == 4
    assert benchmark["education"] == "Bachelor's degree"
    assert benchmark["retrieved_at"] and benchmark["source"]
    assert benchmark["unavailable_fields"] == []
    assert FakeOnetClient.calls[0].endswith("/online/search")


def test_benchmark_partial_failure_records_unavailable(monkeypatch):
    provider = _use_onet(monkeypatch, fail_on={"details/skills"})
    benchmark = provider.benchmark("Data Scientist")
    assert "core_skills" in benchmark["unavailable_fields"]
    assert benchmark["benchmark_quality"] < 1.0
    assert not benchmark["core_skills"]


def test_benchmark_total_failure_never_fabricates(monkeypatch):
    provider = _use_onet(monkeypatch, fail_on={"/online/search"})
    benchmark = provider.benchmark("Data Scientist")
    assert benchmark["onet_code"] is None
    assert not benchmark["core_skills"]
    assert any("failed" in field for field in benchmark["unavailable_fields"])
    assert benchmark["title"] == "Data Scientist"  # only what the caller provided


def test_benchmark_no_match(monkeypatch):
    from fixtures import _FakeResponse
    provider = _use_onet(monkeypatch, responses={
        "/online/search": _FakeResponse({"occupation": []})})
    benchmark = provider.benchmark("Zzzz")
    assert benchmark["onet_code"] is None
    assert any("no O*NET occupation matched" in field for field in benchmark["unavailable_fields"])


def test_empty_benchmark_when_key_missing(monkeypatch):
    monkeypatch.delenv("ONET_API_KEY", raising=False)
    benchmark = onet.provider().benchmark("Data Scientist")
    assert benchmark["provider"] == "none"
    assert set(benchmark["unavailable_fields"]) == set(onet.EmptyBenchmark.UNAVAILABLE)
    assert not benchmark["core_skills"]


def test_importance_to_level_scale():
    assert onet.importance_to_level(90) == 4
    assert onet.importance_to_level(75) == 3
    assert onet.importance_to_level(55) == 2
    assert onet.importance_to_level(35) == 1
    assert onet.importance_to_level(10) == 0
    assert onet.importance_to_level(4.4) == 4  # legacy 1-5 scale normalized


# --- deterministic scoring ---------------------------------------------------------


def _analyze(raw, benchmark=None, as_of=AS_OF):
    profile = pf.normalize_input(raw, provenance="linkedin", as_of=as_of)["profile"]
    return profile, scoring.analyze(profile, benchmark or DS_BENCHMARK, as_of=as_of)


def test_scoring_is_deterministic_and_bounded():
    _, first = _analyze(experienced_direct())
    _, second = _analyze(experienced_direct())
    assert first == second, "identical normalized inputs must reproduce identical scores"
    for key in ("readiness_score", "transferability_score", "evidence_strength_score"):
        assert 0 <= first[key] <= 100
    assert first["transition_difficulty"] in ("low", "medium", "high")
    assert 0 <= first["confidence"] <= 1


def test_rubric_weights_sum_to_one():
    assert abs(sum(scoring.ScoringConfig.weights.values()) - 1.0) < 1e-9
    assert scoring.ScoringConfig.weights["relevant_experience"] == 0.30
    assert scoring.ScoringConfig.weights["skill_coverage"] == 0.25


def test_breakdown_covers_all_categories():
    _, analysis = _analyze(experienced_direct())
    assert set(analysis["breakdown"]) == set(scoring.ScoringConfig.weights)
    for category in analysis["breakdown"].values():
        assert 0 <= category["score"] <= 100
        assert category["evidence"] or category["assumptions"] or category["gaps"]


def test_demonstrated_evidence_outweighs_mentioned():
    strong = json.loads(experienced_direct())
    weak = json.loads(weak_resume_evidence())
    _, strong_analysis = _analyze(strong)
    _, weak_analysis = _analyze(weak)
    assert strong_analysis["breakdown"]["skill_coverage"]["score"] > \
        weak_analysis["breakdown"]["skill_coverage"]["score"]
    assert strong_analysis["evidence_strength_score"] > weak_analysis["evidence_strength_score"]


def test_readiness_ordering_across_archetypes():
    _, direct = _analyze(experienced_direct())
    _, transfer = _analyze(transferable_strong())
    _, intermediate = _analyze(needs_intermediate())
    _, thin = _analyze(career_changer_thin())
    assert direct["readiness_score"] > intermediate["readiness_score"]
    assert transfer["readiness_score"] > thin["readiness_score"]
    assert thin["transition_difficulty"] == "high"
    assert direct["transition_difficulty"] in ("low", "medium")


def test_missing_information_lowers_confidence_and_documents_assumptions():
    profile, analysis = _analyze(incomplete_dates())
    _, complete = _analyze(experienced_direct())
    assert analysis["confidence"] < complete["confidence"]
    assert any("date" in assumption.lower() for assumption in analysis["assumptions"])
    assert analysis["breakdown"]["relevant_experience"]["assumptions"], \
        "undated roles are documented in the category breakdown"
    assert profile["warnings"], "incomplete dates produce warnings, not crashes"


def test_empty_benchmark_degrades_with_assumptions():
    empty = onet.EmptyBenchmark().benchmark("Data Scientist")
    _, analysis = _analyze(experienced_direct(), benchmark=empty)
    _, full = _analyze(experienced_direct())  # same profile against a complete benchmark
    assert any("Benchmark" in a or "unavailable" in a.lower() for a in analysis["assumptions"])
    assert analysis["gaps"] == []
    assert analysis["confidence"] < full["confidence"], "missing benchmark data lowers confidence"


# --- gap analysis --------------------------------------------------------------------


def test_gap_schema_and_types():
    _, analysis = _analyze(experienced_direct())
    for gap in analysis["gaps"]:
        assert set(gap) >= {"skill", "current_level", "required_level", "gap_type", "priority", "evidence", "source"}
        assert gap["gap_type"] in ("confirmed_gap", "missing_evidence", "unverified")
        assert 0 <= gap["current_level"] <= 4 and 1 <= gap["required_level"] <= 4
        assert gap["priority"] in ("high", "medium", "low")
        assert gap["source"]


def test_gap_type_distinguishes_missing_evidence_from_confirmed_gap():
    _, thin = _analyze(career_changer_thin())
    thin_types = {g["gap_type"] for g in thin["gaps"]}
    assert "confirmed_gap" not in thin_types, "a thin profile must not claim confirmed inability"
    assert "missing_evidence" in thin_types

    _, strong = _analyze(experienced_direct())
    strong_types = {g["gap_type"] for g in strong["gaps"]}
    assert "confirmed_gap" in strong_types, "a detailed profile with a missing core skill shows a real gap"


def test_unverified_for_listed_only_skills():
    _, analysis = _analyze(weak_resume_evidence())
    unverified = [g for g in analysis["gaps"] if g["gap_type"] == "unverified"]
    assert unverified, "skills-list-only evidence must stay unverified, not confirmed"


def test_gap_priorities_ordered():
    _, analysis = _analyze(needs_intermediate())
    priorities = [g["priority"] for g in analysis["gaps"]]
    assert priorities == sorted(priorities, key={"high": 0, "medium": 1, "low": 2}.get)


# --- trajectories ----------------------------------------------------------------------


def test_plan_generates_and_compares_options():
    profile, analysis = _analyze(experienced_direct())
    benchmark = DS_BENCHMARK
    plan_data = trajectories.plan(profile, analysis, benchmark)
    assert 2 <= len(plan_data["paths"]) <= 3
    assert plan_data["recommended_id"] in {p["id"] for p in plan_data["paths"]}
    best = max(plan_data["comparison"], key=lambda row: row["composite"])["id"]
    assert best == plan_data["recommended_id"]
    for path in plan_data["paths"]:
        assert path["type"] in (
            trajectories.PATH_TYPE_DIRECT, trajectories.PATH_TYPE_BRIDGE,
            trajectories.PATH_TYPE_INTERMEDIATE, trajectories.PATH_TYPE_RESKILL)
        assert path["roles"] and path["rationale"] and path["estimated_months"] >= 3
        assert path["waypoints"] and path["risks"] and path["confidence"] <= 0.95
        assert "guarantee" in " ".join(path["risks"]).lower()


def test_plan_types_match_readiness():
    profile, thin = _analyze(career_changer_thin())
    plan_data = trajectories.plan(profile, thin, DS_BENCHMARK)
    assert plan_data["paths"][0]["type"] == trajectories.PATH_TYPE_RESKILL
    profile, direct = _analyze(experienced_direct())
    direct_plan = trajectories.plan(profile, direct, DS_BENCHMARK)
    assert direct_plan["paths"][0]["type"] == trajectories.PATH_TYPE_DIRECT


def test_select_marks_and_keeps_alternatives():
    profile, analysis = _analyze(experienced_direct())
    plan_data = trajectories.plan(profile, analysis, DS_BENCHMARK)
    selection = trajectories.select(plan_data, analysis, plan_data["recommended_id"])
    assert selection["trajectory"]["phases"]
    assert selection["scores"]["readiness"] == analysis["readiness_score"]
    assert len(selection["alternatives"]) == len(plan_data["paths"]) - 1
    assert trajectories.select(plan_data, analysis, "path-99")["error"]


def test_payload_schema_and_validation():
    profile, analysis = _analyze(experienced_direct())
    plan_data = trajectories.plan(profile, analysis, DS_BENCHMARK)
    selection = trajectories.select(plan_data, analysis, plan_data["recommended_id"])
    payload = trajectories.build_payload(profile, analysis, selection)
    assert payload["schema_version"] == "2.0"
    assert payload["trajectory_id"].startswith("traj_")
    assert payload["created_at"]
    assert "name" not in payload and "Maya Chen" not in json.dumps(payload), "no identity data downstream"
    waypoints = payload["waypoints"]
    assert waypoints
    assert all({"outcome", "target_horizon", "success_criteria"} <= set(w) for w in waypoints)
    assert trajectories.validate_payload(payload) == []
    assert payload["needs_user_confirmation"] == [] or isinstance(payload["needs_user_confirmation"], list)


def test_payload_validation_rejects_bad_payloads():
    base = {"schema_version": "2.0", "trajectory_id": "traj_x", "profile_id": "anon-1",
            "target_role": "DS", "scores": {"readiness": 50, "transferability": 50,
                                            "evidence_strength": 50, "transition_difficulty": "medium"},
            "selected_path": {"title": "t", "roles": [], "estimated_months": 6, "rationale": "r"},
            "waypoints": [{"outcome": "o", "target_horizon": "0-3 months", "success_criteria": ["x"]}],
            "confidence": 0.5, "created_at": "2026-06-01T00:00:00+00:00"}
    assert trajectories.validate_payload(base) == []
    broken = [{**base, "schema_version": "1.0"},
              {**base, "scores": {**base["scores"], "readiness": 150}},
              {**base, "scores": {**base["scores"], "transition_difficulty": "huge"}},
              {**base, "confidence": 2},
              {**base, "created_at": "not-a-date"},
              {**base, "waypoints": []},
              {**base, "daily_schedule": [{"day": "monday"}]},
              {**base, "skill_gaps": [{"skill": "X", "gap_type": "made_up"}]}]
    for payload in broken:
        assert trajectories.validate_payload(payload), f"must reject {payload}"


# --- what-if scenarios -----------------------------------------------------------------


def test_what_if_compares_scenarios_without_new_pipeline():
    profile, _ = _analyze(experienced_direct())
    scenarios = [
        {"type": "add_certification", "value": "TensorFlow Developer Certificate"},
        {"type": "remote_only"},
        {"type": "accelerated"},
        {"type": "alternate_role", "role": "Data Engineer", "label": "switch to Data Engineer"},
    ]
    benchmarks = {"Data Scientist": DS_BENCHMARK,
                  "Data Engineer": dict(DS_BENCHMARK, role="Data Engineer", title="Data Engineers",
                                        job_zone=3)}
    result = scoring.what_if(profile, benchmarks, scenarios, as_of=AS_OF)
    rows = result["rows"]
    assert rows[0]["scenario"].startswith("current plan")
    assert rows[0]["delta"] == 0
    assert len(rows) == 1 + len(scenarios)
    cert_row = rows[1]
    assert cert_row["delta"] is not None and abs(cert_row["delta"]) <= 20  # 5% weight caps the effect
    assert rows[-1]["target_role"] == "Data Engineer"
    assert result["disclaimer"]


def test_what_if_unknown_role_reports_without_crashing():
    profile, _ = _analyze(experienced_direct())
    result = scoring.what_if(profile, {"Data Scientist": DS_BENCHMARK},
                             [{"type": "alternate_role", "role": "Unheard Role"}], as_of=AS_OF)
    assert result["rows"][-1]["readiness"] is None
    assert "No benchmark fetched" in result["rows"][-1]["notes"]


def test_multiple_target_roles_scored_independently():
    _, ds = _analyze(experienced_direct(), benchmark=DS_BENCHMARK)
    engineer_benchmark = dict(DS_BENCHMARK, role="Data Engineer", title="Data Engineers",
                              core_skills=[{"name": "SQL", "importance": 90, "level": 90,
                                            "required_level": 4},
                                           {"name": "Python", "importance": 80, "level": 80,
                                            "required_level": 3}])
    _, engineer = _analyze(experienced_direct(), benchmark=engineer_benchmark)
    assert ds["readiness_score"] != engineer["readiness_score"] or \
        {g["skill"] for g in ds["gaps"]} != {g["skill"] for g in engineer["gaps"]}
