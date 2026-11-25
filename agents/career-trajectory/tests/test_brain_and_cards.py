"""Brain tools, card builders, and the Goal Planning handoff.

Never calls the real ASI:One API, Agentverse, or O*NET: tools are pure or run
against a fake httpx client, and the handoff uses a stub agent Context.
"""

import json
import os

from fixtures import (
    AS_OF,
    DS_BENCHMARK,
    FakeOnetClient,
    experienced_direct,
    onet_default_responses,
    plain_resume_text,
)
from uagents_core.contrib.protocols.chat import EndSessionContent, MetadataContent, TextContent

import career_profile as pf
import scoring
import trajectories

os.environ.setdefault("AGENT_SEED", "tests-only-seed-never-used-for-a-real-agent")

import agent
import brain
import cards

SAMPLE_HISTORY = {
    "name": "Maya Chen",
    "headline": "Senior Data Analyst",
    "skills": ["SQL", "Python", "Tableau"],
    "experiences": [
        {"title": "Data Analyst", "company": "Acme Labs", "start": "2019-07", "end": "2021-07",
         "skills": ["SQL", "Excel"]},
        {"title": "Senior Data Analyst", "company": "Acme Labs", "start": "2021-07", "end": "2023-07",
         "skills": ["SQL", "Python"]},
    ],
}


def _invoke_tool(tool_obj, **kwargs):
    return json.loads(tool_obj.invoke(kwargs))


# --- parse_history ----------------------------------------------------------


def test_parse_history_json_facts():
    data = _invoke_tool(brain.parse_history, raw=json.dumps(SAMPLE_HISTORY))
    assert "error" not in data
    facts = data["facts"]
    assert facts["Years of experience"] == "4.0"
    assert facts["Roles held"] == "2"
    assert facts["Average tenure"] == "2.0 years"
    assert facts["Current role"] == "Senior Data Analyst"
    assert facts["Latest employer"] == "Acme Labs"
    assert facts["Progression"] == "Data Analyst -> Senior Data Analyst"
    assert facts["Top skills"].startswith("SQL")
    assert data["title"].startswith("Maya Chen")


def test_parse_history_with_target_role():
    payload = dict(SAMPLE_HISTORY, target_role="Data Scientist")
    data = _invoke_tool(brain.parse_history, raw=json.dumps(payload))
    assert data["target_role"] == "Data Scientist"


def test_parse_history_provenance_and_canonical_profile():
    data = _invoke_tool(brain.parse_history, raw=json.dumps(SAMPLE_HISTORY))
    profile = data["profile"]
    assert profile["schema_version"] == "2.0"
    assert profile["experiences"][0]["provenance"] == ["linkedin"]
    assert profile["profile_id"].startswith("anon-")
    free = _invoke_tool(brain.parse_history, raw="Data Analyst, Acme Labs\n2019 - 2021")
    assert free["profile"]["experiences"][0]["provenance"] == ["user"]


def test_parse_history_partial_json():
    raw = json.dumps({"experiences": [{"title": "Teacher", "company": "Lincoln High", "start": "2018", "end": "2022"}]})
    data = _invoke_tool(brain.parse_history, raw=raw)
    assert "error" not in data
    assert data["facts"]["Roles held"] == "1"
    assert data["facts"]["Years of experience"] == "4.0"


def test_parse_history_free_text():
    raw = "Data Analyst at Acme Labs 2019-2021, then Senior Data Analyst at Acme Labs 2021-present"
    data = _invoke_tool(brain.parse_history, raw=raw)
    assert "error" not in data
    assert int(data["facts"]["Roles held"]) >= 1


def test_parse_history_garbage():
    data = _invoke_tool(brain.parse_history, raw="lorem ipsum dolor sit amet")
    assert "error" in data


# --- lookup_occupation (benchmark) -------------------------------------------


def _use_fake_onet(monkeypatch, responses=None, fail_on=()):
    import onet
    monkeypatch.setenv("ONET_API_KEY", "test-key")
    monkeypatch.setattr(onet.httpx, "Client", FakeOnetClient)
    FakeOnetClient.calls = []
    FakeOnetClient.fail_on = set(fail_on)
    FakeOnetClient.responses = responses if responses is not None else onet_default_responses()


def test_lookup_occupation_benchmark(monkeypatch):
    _use_fake_onet(monkeypatch)
    data = _invoke_tool(brain.lookup_occupation, role="Data Scientist")
    assert "error" not in data
    assert data["title"] == "Data Scientists"
    assert data["facts"]["O*NET code"] == "15-2051.00"
    assert data["facts"]["Typical skills"].startswith("Python")
    assert data["facts"]["Job zone"] == "4"
    assert data["facts"]["Typical education"] == "Bachelor's degree"
    assert "Benchmark retrieved" in data["facts"]
    benchmark = data["benchmark"]
    assert benchmark["core_skills"][0]["required_level"] >= 3
    assert benchmark["retrieved_at"]
    assert FakeOnetClient.calls[0].endswith("/online/search")


def test_lookup_occupation_search_failure(monkeypatch):
    _use_fake_onet(monkeypatch, fail_on={"/online/search"})
    data = _invoke_tool(brain.lookup_occupation, role="Data Scientist")
    assert "error" in data
    assert data["benchmark"]["unavailable_fields"]


def test_lookup_occupation_no_match(monkeypatch):
    from fixtures import _FakeResponse
    _use_fake_onet(monkeypatch, responses={"/online/search": _FakeResponse({"occupation": []})})
    data = _invoke_tool(brain.lookup_occupation, role="Zzzz")
    assert "error" in data


def test_lookup_occupation_missing_key(monkeypatch):
    monkeypatch.delenv("ONET_API_KEY", raising=False)
    data = _invoke_tool(brain.lookup_occupation, role="Data Scientist")
    assert "warning" in data
    assert "not configured" in data["facts"]["Benchmark source"]


# --- analyze_readiness --------------------------------------------------------


def _profile_json(raw=None):
    result = pf.normalize_input(raw or json.loads(experienced_direct()), provenance="linkedin", as_of=AS_OF)
    return json.dumps(result["profile"])


def test_analyze_readiness_full():
    data = _invoke_tool(brain.analyze_readiness, profile_json=_profile_json(),
                        benchmark_json=json.dumps(DS_BENCHMARK))
    assert "error" not in data
    assert data["facts"]["Readiness"].endswith("/100")
    analysis = data["analysis"]
    assert 0 <= analysis["readiness_score"] <= 100
    assert analysis["target_role"] == "Data Scientist"
    assert set(analysis["breakdown"]) == set(scoring.ScoringConfig.weights)
    assert analysis["confidence"] <= 0.95


def test_analyze_readiness_needs_target_role():
    history = dict(SAMPLE_HISTORY)
    history.pop("target_role", None)
    data = _invoke_tool(brain.analyze_readiness, profile_json=_profile_json(history),
                        benchmark_json="{}")
    assert data["needs_input"]["field"] == "target_role"
    assert "error" not in data


def test_analyze_readiness_empty_benchmark_degrades():
    data = _invoke_tool(brain.analyze_readiness, profile_json=_profile_json(), benchmark_json="{}")
    assert data["analysis"]["gaps"] == []
    assert any("unavailable" in a.lower() or "benchmark" in a.lower() for a in data["analysis"]["assumptions"])


def test_analyze_readiness_merges_resume_fragment():
    profile = json.loads(_profile_json())
    fragment = pf.normalize_input(plain_resume_text(), provenance="resume", as_of=AS_OF)["profile"]
    data = _invoke_tool(brain.analyze_readiness, profile_json=json.dumps(profile),
                        benchmark_json=json.dumps(DS_BENCHMARK), resume_json=json.dumps(fragment))
    assert "error" not in data
    projects = data["profile"].get("projects") or []
    assert projects, "résumé fragment projects are merged into the unified profile"


def test_analyze_readiness_surfaces_conflicts():
    linkedin = pf.normalize_input(
        {"name": "Riley", "experiences": [{"title": "Data Analyst", "company": "Acme Labs",
                                           "start": "2021-01", "end": "2023-01"}]},
        provenance="linkedin", as_of=AS_OF)["profile"]
    fragment = pf.normalize_input(
        {"experiences": [{"title": "Analytics Consultant", "company": "Acme Labs",
                          "start": "2020-06", "end": "2023-01"}]},
        provenance="resume", as_of=AS_OF)["profile"]
    data = _invoke_tool(brain.analyze_readiness, profile_json=json.dumps(linkedin),
                        benchmark_json=json.dumps(DS_BENCHMARK), resume_json=json.dumps(fragment))
    assert data["analysis"]["conflicts"], "conflicting facts are flagged, never silently dropped"
    assert data["analysis"]["needs_user_confirmation"]


# --- plan_trajectories / choose_path -------------------------------------------


def _analysis_json(raw=None):
    profile = json.loads(_profile_json(raw))
    return json.dumps(scoring.analyze(profile, DS_BENCHMARK, as_of=AS_OF)), profile


def test_plan_trajectories_tool():
    analysis, profile = _analysis_json()
    data = _invoke_tool(brain.plan_trajectories, analysis_json=analysis,
                        profile_json=json.dumps(profile), benchmark_json=json.dumps(DS_BENCHMARK))
    assert len(data["paths"]) in (2, 3)
    assert data["recommended_id"] == data["plan"]["recommended_id"]
    for path in data["paths"]:
        assert {"id", "title", "summary", "skill_gaps"} <= set(path)
    assert data["plan"]["disclaimer"]


def test_plan_trajectories_requires_analysis():
    data = _invoke_tool(brain.plan_trajectories, analysis_json="{}", profile_json="{}",
                        benchmark_json="{}")
    assert "error" in data


def test_choose_path_tool():
    analysis, profile = _analysis_json()
    planned = _invoke_tool(brain.plan_trajectories, analysis_json=analysis,
                           profile_json=json.dumps(profile), benchmark_json=json.dumps(DS_BENCHMARK))
    data = _invoke_tool(brain.choose_path, plan_json=json.dumps(planned["plan"]),
                        analysis_json=analysis, path_id=planned["recommended_id"])
    assert data["trajectory"]["phases"]
    assert data["trajectory"]["title"]
    assert data["scores"]["readiness"] == json.loads(analysis)["readiness_score"]
    assert data["alternatives"]
    bad = _invoke_tool(brain.choose_path, plan_json=json.dumps(planned["plan"]),
                       analysis_json=analysis, path_id="path-9")
    assert "error" in bad


# --- what_if ---------------------------------------------------------------------


def test_what_if_tool():
    _, profile = _analysis_json()
    benchmarks = {"Data Scientist": DS_BENCHMARK,
                  "Data Engineer": dict(DS_BENCHMARK, role="Data Engineer", title="Data Engineers")}
    data = _invoke_tool(brain.what_if, profile_json=json.dumps(profile),
                        benchmarks_json=json.dumps(benchmarks),
                        scenarios_json=json.dumps([
                            {"type": "add_certification", "value": "TensorFlow Developer Certificate"},
                            {"type": "alternate_role", "role": "Data Engineer"}]))
    assert data["title"] == "What-if comparison"
    assert "current plan (baseline)" in data["facts"]
    assert data["what_if"]["rows"][0]["delta"] == 0
    bad = _invoke_tool(brain.what_if, profile_json=json.dumps(profile),
                       benchmarks_json=json.dumps(benchmarks), scenarios_json="[]")
    assert "error" in bad


# --- cards.normalize_incoming ------------------------------------------------


def test_normalize_selections():
    assert cards.normalize_incoming('{"action": "pick", "id": "path-2"}') == "select path path-2"
    assert cards.normalize_incoming('{"action": "new_query"}') == "start a new trajectory"
    assert cards.normalize_incoming('{"action": "plan_goals"}') == cards.PLAN_GOALS
    out = cards.normalize_incoming('{"target_role": "Data Scientist", "action": "submit"}')
    assert out == "target role: Data Scientist"
    assert cards.normalize_incoming('{"query": "Paris", "action": "submit"}') == "target role: Paris"


def test_normalize_prose_and_mentions():
    text = "The user submitted query Paris"
    assert cards.normalize_incoming(text) == text
    assert cards.normalize_incoming("@pathfinder What can you do?") == "What can you do?"


def test_normalize_passthrough_history_json():
    history = '{"name": "Maya", "experiences": [{"title": "Data Analyst"}]}'
    assert cards.normalize_incoming(history) == history


# --- cards.build_reply --------------------------------------------------------


def _metas(message):
    found = [item for item in message.content if isinstance(item, MetadataContent)]
    assert len(found) == 1, f"expected exactly one MetadataContent, got {len(found)}"
    return found[0].metadata


def _text(message):
    found = [item for item in message.content if isinstance(item, TextContent)]
    assert len(found) == 1
    return found[0].text


def test_build_reply_detail_card():
    message = cards.build_reply("Profile ready.", {"title": "Maya Chen", "facts": {"Years of experience": "4.0"}})
    meta = _metas(message)
    assert meta["card_kind"] == "detail"
    assert meta["card_protocol_version"] == "1"
    assert meta["is_terminal"] == "true"
    assert all(isinstance(value, str) for value in meta.values())
    payload = json.loads(meta["card_payload"])
    assert payload["title"] == "Maya Chen"
    assert payload["summary_rows"][0] == {"label": "Years of experience", "value": "4.0"}
    assert _text(message) == "Profile ready."
    assert any(isinstance(item, EndSessionContent) for item in message.content)


def test_build_reply_carousel_card():
    data = {
        "title": "Career paths toward Data Scientist",
        "recommended_id": "path-1",
        "paths": [
            {"id": "path-1", "title": "Direct transition", "summary": "Ready now.", "skill_gaps": ["ML"]},
            {"id": "path-2", "title": "Skill bridge", "summary": "Close two gaps.", "skill_gaps": []},
        ],
    }
    message = cards.build_reply("Two options stand out.", data)
    meta = _metas(message)
    assert meta["card_kind"] == "carousel"
    assert "is_terminal" not in meta
    payload = json.loads(meta["card_payload"])
    assert payload["items"][0]["primary_cta"]["selection"] == {"action": "pick", "id": "path-1"}
    assert payload["items"][0]["subtitle"].startswith("Recommended.")
    assert all(isinstance(value, str) for value in meta.values())
    assert not any(isinstance(item, EndSessionContent) for item in message.content)


def test_build_reply_trajectory_card():
    data = {
        "target_role": "Data Scientist",
        "scores": {"readiness": 68, "transferability": 84, "transition_difficulty": "medium"},
        "trajectory": {
            "title": "Trajectory toward Data Scientist",
            "phases": [{"n": 1, "goal": "Learn statistics", "horizon": "0-3 months"}],
            "skill_gaps": ["Statistics"],
            "risks": ["Estimates are planning guidance"],
            "estimated_months": 12,
            "confidence": 0.81,
        },
    }
    message = cards.build_reply("Here's your path.", data)
    meta = _metas(message)
    assert meta["card_kind"] == "detail"
    assert "is_terminal" not in meta  # has CTAs, so it must accept input
    payload = json.loads(meta["card_payload"])
    actions = [cta["selection"]["action"] for cta in payload["ctas"]]
    assert actions == ["plan_goals", "new_query"]
    assert payload["summary_rows"][0]["value"] == "Learn statistics"
    assert payload["summary_rows"][-1] == {"label": "Target role", "value": "Data Scientist"}
    rows = {row["label"]: row["value"] for row in payload["summary_rows"]}
    assert rows["Confidence"] == "81%"
    assert rows["Estimated horizon"].startswith("about 12 months")


def test_build_reply_form_card():
    data = {"needs_input": {"field": "target_role", "label": "What target role?"}}
    message = cards.build_reply("Which role should we aim for?", data)
    meta = _metas(message)
    assert meta["card_kind"] == "form"
    payload = json.loads(meta["card_payload"])
    assert payload["fields"][0]["required"] is True
    assert payload["submit_cta"]["selection"] == {"action": "submit"}


def test_build_reply_error_text_only():
    message = cards.build_reply("That failed.", {"error": "upstream down"})
    assert _text(message) == "That failed."
    assert not [item for item in message.content if isinstance(item, MetadataContent)]
    assert any(isinstance(item, EndSessionContent) for item in message.content)


def test_build_reply_no_data_text_only():
    message = cards.build_reply("Hello!", None)
    assert _text(message) == "Hello!"
    assert not [item for item in message.content if isinstance(item, MetadataContent)]
    assert any(isinstance(item, EndSessionContent) for item in message.content)


def test_card_payloads_stay_under_64k():
    big = {"title": "Big", "facts": {f"Fact {i}": "x" * 5000 for i in range(20)}}
    message = cards.build_reply("Big profile.", big)
    meta = _metas(message)
    assert len(meta["card_payload"]) < 65536
    assert json.loads(meta["card_payload"])  # still valid JSON


# --- Goal Planning handoff (agent) ----------------------------------------------


class _StubCtx:
    def __init__(self, fail_forward_to=None):
        self.sent: list = []
        self.store: dict = {}
        self.fail_forward_to = fail_forward_to

        class _Logger:
            def info(self, message):
                pass

            def error(self, message):
                pass

            def exception(self, message):
                pass

        self.logger = _Logger()

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    async def send(self, target, message):
        if target == self.fail_forward_to:
            raise RuntimeError("mailbox down")
        self.sent.append((target, message))


def _handoff_state():
    result = pf.normalize_input(json.loads(experienced_direct()), provenance="linkedin", as_of=AS_OF)
    profile = result["profile"]
    analysis = scoring.analyze(profile, DS_BENCHMARK, as_of=AS_OF)
    plan_data = trajectories.plan(profile, analysis, DS_BENCHMARK)
    selection = trajectories.select(plan_data, analysis, plan_data["recommended_id"])
    return {"profile": profile, "analysis": analysis, "selection": selection}


def test_handoff_emits_valid_payload():
    ctx = _StubCtx()
    import asyncio
    asyncio.run(agent._hand_off(ctx, "user_address", _handoff_state()))
    assert len(ctx.sent) == 1
    text = ctx.sent[0][1].content[0].text
    assert "goal-planning payload" in text
    payload_text = text.split("payload:\n\n", 1)[1].rsplit("\n", 1)[0]
    payload = json.loads(payload_text)
    assert trajectories.validate_payload(payload) == []
    assert payload["trajectory_id"].startswith("traj_")


def test_handoff_without_trajectory_guides_user():
    ctx = _StubCtx()
    import asyncio
    asyncio.run(agent._hand_off(ctx, "user_address", {}))
    text = ctx.sent[0][1].content[0].text
    assert "selected trajectory" in text


def test_handoff_rejects_invalid_payload():
    ctx = _StubCtx()
    state = _handoff_state()
    state["selection"]["scores"] = {}  # corrupt the scores -> validation must fail
    import asyncio
    asyncio.run(agent._hand_off(ctx, "user_address", state))
    text = ctx.sent[0][1].content[0].text
    assert "failed validation" in text
    assert len(ctx.sent) == 1  # nothing forwarded


def test_handoff_forwards_and_fails_gracefully(monkeypatch):
    import asyncio
    monkeypatch.setattr(agent, "AGENT3_ADDRESS", "agent3-address")
    ctx = _StubCtx()
    asyncio.run(agent._hand_off(ctx, "user_address", _handoff_state()))
    targets = [target for target, _ in ctx.sent]
    assert "agent3-address" in targets and "user_address" in targets
    assert "on its way" in ctx.sent[-1][1].content[0].text

    failing = _StubCtx(fail_forward_to="agent3-address")
    asyncio.run(agent._hand_off(failing, "user_address", _handoff_state()))
    assert "copy the JSON" in failing.sent[-1][1].content[0].text
    assert all(target == "user_address" for target, _ in failing.sent)


def test_merge_state_rebuilds_selection_from_choose_path_data():
    """Regression: choose_path flattens its selection (title/selected/scores/trajectory/
    alternatives) at the top level of the tool result; _merge_state must rebuild the
    nested "selection" so the Plan-my-goals handoff works after a card tap."""
    result = pf.normalize_input(json.loads(experienced_direct()), provenance="linkedin", as_of=AS_OF)
    profile = result["profile"]
    analysis = scoring.analyze(profile, DS_BENCHMARK, as_of=AS_OF)
    plan_data = trajectories.plan(profile, analysis, DS_BENCHMARK)
    selection = trajectories.select(plan_data, analysis, plan_data["recommended_id"])
    tool_data = {**selection, "plan": plan_data, "analysis": analysis,
                 "profile": profile, "benchmark": DS_BENCHMARK}

    state = agent._merge_state({}, tool_data)
    assert state["selection"]["selected"] == plan_data["recommended_id"]
    assert state["selection"]["trajectory"]["phases"]
    assert state["selection"]["scores"]["readiness"] == analysis["readiness_score"]
    assert state["selection"]["type"] == plan_data["paths"][0]["type"]

    import asyncio
    ctx = _StubCtx()
    asyncio.run(agent._hand_off(ctx, "user_address", state))
    assert ctx.sent and "goal-planning payload" in ctx.sent[0][1].content[0].text
    payload_text = ctx.sent[0][1].content[0].text.split("payload:\n\n", 1)[1].rsplit("\n", 1)[0]
    assert trajectories.validate_payload(json.loads(payload_text)) == []


def test_select_carries_path_type():
    result = pf.normalize_input(json.loads(experienced_direct()), provenance="linkedin", as_of=AS_OF)
    profile = result["profile"]
    analysis = scoring.analyze(profile, DS_BENCHMARK, as_of=AS_OF)
    plan_data = trajectories.plan(profile, analysis, DS_BENCHMARK)
    selection = trajectories.select(plan_data, analysis, plan_data["paths"][0]["id"])
    assert selection["type"] == plan_data["paths"][0]["type"]
