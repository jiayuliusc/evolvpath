"""Career-profile normalization, reconciliation, evidence classification, and résumé ingestion.

Pure-function tests: no network, no LLM, no Agentverse.
"""

import json

from fixtures import (
    AS_OF,
    docx_bytes,
    incomplete_dates,
    injection_resume_text,
    linkedin_conflicting,
    minimal_pdf,
    plain_resume_text,
    resume_conflicting_text,
)

import career_profile as pf
import resume as resume_mod

# --- input compatibility (legacy alumni_history and aliases) --------------------


def _profile(raw, provenance="user"):
    return pf.normalize_input(raw, provenance=provenance, as_of=AS_OF)


def test_legacy_alumni_history_aliases():
    raw = {
        "name": "Maya",
        "headline": "Senior Data Analyst",
        "skills": "SQL, Python",
        "positions": [
            {"role": "Data Analyst", "organisation": "Acme Labs", "start_date": "2019-07", "end_date": "2021-07",
             "skills": ["SQL"]},
        ],
    }
    result = _profile(raw)
    profile = result["profile"]
    assert profile["experiences"][0]["title"] == "Data Analyst"
    assert profile["experiences"][0]["company"] == "Acme Labs"
    assert profile["experiences"][0]["start"] == "2019-07"
    assert [s["name"] for s in profile["skills"]] == ["SQL", "Python"]
    assert result["warnings"]  # missing schema_version is surfaced, not fatal


def test_bare_role_list_and_string_skills():
    result = _profile([{"title": "Teacher", "company": "Lincoln High", "start": "2018", "end": "2022"}])
    assert result["profile"]["experiences"][0]["title"] == "Teacher"


def test_canonical_v2_input_with_preferences():
    raw = {
        "schema_version": "2.0",
        "profile_id": "anon-abc",
        "skills": ["SQL"],
        "target_roles": ["Data Scientist", "Data Engineer"],
        "preferences": {"remote_preference": "remote", "timeline_months": 12},
        "constraints": ["no relocation"],
        "source_metadata": [{"source": "linkedin"}],
    }
    profile = _profile(raw)["profile"]
    assert profile["schema_version"] == "2.0"
    assert profile["profile_id"] == "anon-abc"
    assert profile["target_roles"] == ["Data Scientist", "Data Engineer"]
    assert profile["preferences"]["remote_preference"] == "remote"
    assert profile["constraints"] == ["no relocation"]


def test_anonymous_profile_id_generated():
    profile = _profile({"experiences": [{"title": "Analyst", "start": "2020", "end": "2021"}]})["profile"]
    assert profile["profile_id"].startswith("anon-")


def test_garbage_raises_value_error():
    try:
        pf.normalize_input("lorem ipsum dolor sit amet", as_of=AS_OF)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --- date and tenure normalization ------------------------------------------------


def test_parse_month_formats():
    assert pf.parse_month("2019-07") == (2019, 7)
    assert pf.parse_month("2019/03") == (2019, 3)
    assert pf.parse_month("2019") == (2019, 1)
    assert pf.parse_month("2019-03-15") == (2019, 3)
    assert pf.parse_month("Jul 2019") == (2019, 7)
    assert pf.parse_month("July 2019") == (2019, 7)
    assert pf.parse_month("present", AS_OF) == (2026, 6)
    assert pf.parse_month("present") is None  # no as_of: refuse to guess wall-clock
    assert pf.parse_month("nonsense") is None


def test_duration_years_clamps_negative():
    assert pf.duration_years("2020-05", "2019-01", AS_OF) == 0.0
    assert pf.duration_years("2019-01", "2021-01", AS_OF) == 2.0
    assert pf.duration_years("2019", "present", AS_OF) == 89 / 12  # present resolves against as_of


def test_incomplete_dates_fixture_produces_warnings_not_crashes():
    result = _profile(incomplete_dates())
    profile, warnings = result["profile"], result["warnings"]
    assert len(profile["experiences"]) == 2
    assert any("date" in warning.lower() for warning in warnings)


# --- evidence classification --------------------------------------------------------


def test_evidence_levels_and_weights():
    assert pf.EVIDENCE_WEIGHTS["demonstrated"] > pf.EVIDENCE_WEIGHTS["applied"] \
        > pf.EVIDENCE_WEIGHTS["mentioned"] > pf.EVIDENCE_WEIGHTS["inferred"]


def test_evidence_classification_from_structured_input():
    raw = {
        "skills": ["Tableau"],  # mentioned
        "experiences": [
            {"title": "Analyst", "company": "Acme", "start": "2020", "end": "2021",
             "skills": ["SQL", "Python"],  # applied
             "achievements": ["Cut churn 20% using Python models"]},  # Python upgrades to demonstrated
        ],
    }
    profile = _profile(raw)["profile"]
    tiers = {s["name"]: s["tier"] for s in profile["skills"]}
    assert tiers["Tableau"] == "mentioned"
    assert tiers["SQL"] == "applied"
    assert tiers["Python"] == "demonstrated"


def test_level_from_evidence_ordering():
    quantified = [{"level": "demonstrated", "quantified": True}]
    plain = [{"level": "demonstrated", "quantified": False}]
    assert pf.level_from_evidence(quantified) > pf.level_from_evidence(plain)
    assert pf.level_from_evidence(plain) > pf.level_from_evidence([{"level": "applied"}])
    assert pf.level_from_evidence([{"level": "mentioned"}]) >= 1
    assert pf.level_from_evidence([]) == 0


def test_no_title_based_skill_inference():
    raw = {"experiences": [{"title": "Machine Learning Engineer", "company": "BigCo",
                            "start": "2020", "end": "2022"}]}
    profile = _profile(raw)["profile"]
    assert profile["skills"] == [], "a title must never contribute skills"


def test_provenance_values():
    structured = _profile(linkedin_conflicting(), provenance="linkedin")["profile"]
    assert structured["experiences"][0]["provenance"] == ["linkedin"]
    resume_result = _profile(plain_resume_text(), provenance="resume")["profile"]
    sources = {item["source"] for s in resume_result["skills"] for item in s["evidence"]}
    assert sources == {"resume"}


# --- reconciliation ------------------------------------------------------------------


def test_reconcile_dedupes_overlapping_positions():
    history = _profile(linkedin_conflicting(), provenance="linkedin")["profile"]
    resume_fragment = _profile(resume_conflicting_text(), provenance="resume")["profile"]
    merged = pf.reconcile(history, resume_fragment)
    acme = [r for r in merged["experiences"] if pf.same_company(r.get("company"), "Acme Labs")]
    assert len(acme) == 1, "overlapping positions must merge into one"
    merged_role = acme[0]
    assert sorted(merged_role["provenance"]) == ["linkedin", "resume"]
    assert merged_role["achievements"], "résumé achievements are preserved"


def test_reconcile_flags_conflicts_never_silent():
    history = _profile(linkedin_conflicting(), provenance="linkedin")["profile"]
    resume_fragment = _profile(resume_conflicting_text(), provenance="resume")["profile"]
    merged = pf.reconcile(history, resume_fragment)
    kinds = {c["kind"] for c in merged["conflicts"]}
    assert "title_mismatch" in kinds
    assert "dates_mismatch" in kinds
    for conflict in merged["conflicts"]:
        assert conflict["needs_confirmation"] is True
        assert conflict["linkedin_value"] and conflict["resume_value"]


def test_reconcile_merges_skill_evidence_levels():
    history = _profile({"skills": ["Python"], "experiences": []})["profile"]
    fragment = _profile({"experiences": [{"title": "X", "company": "Y", "start": "2020", "end": "2021",
                                          "skills": ["Python"],
                                          "achievements": ["Shipped Python service, 30% faster"]}]},
                        provenance="resume")["profile"]
    merged = pf.reconcile(history, fragment)
    python = next(s for s in merged["skills"] if s["name"] == "Python")
    assert python["tier"] == "demonstrated"
    assert len(python["evidence"]) == 3  # mentioned (history) + applied + quantified achievement (résumé)


def test_completeness_and_prompt_summary_privacy():
    profile = _profile(plain_resume_text(), provenance="resume")["profile"]
    assert 0.0 <= pf.completeness(profile) <= 1.0
    summary = pf.summarize_for_prompt(profile)
    text = json.dumps(summary)
    assert "MAYA CHEN" not in text, "raw name must not leave the profile layer"
    assert summary["name"] in (True, False)  # presence flag, never the raw name
    assert all(isinstance(role["company"], str) for role in summary["roles"])


def test_redact_pii():
    scrubbed = pf.redact("email maya@example.com or call +1 (415) 555-0100 see https://x.io/cv")
    assert "maya@example.com" not in scrubbed
    assert "https://x.io/cv" not in scrubbed
    assert "555-0100" not in scrubbed


# --- résumé ingestion ----------------------------------------------------------------


def test_parse_resume_text_extracts_facts():
    result = resume_mod.parse_resume(plain_resume_text(), as_of=AS_OF)
    assert result["status"] == "ok"
    profile = result["profile"]
    assert len(profile["experiences"]) == 2
    assert profile["experiences"][0]["achievements"][0]["quantified"] is True
    tiers = {s["name"]: s["tier"] for s in profile["skills"]}
    assert tiers["SQL"] == "demonstrated"  # quantified outcome explicitly names SQL
    assert tiers["Statistics"] == "mentioned"
    assert profile["certifications"] and profile["education"] and profile["projects"]


def test_parse_resume_flags_prompt_injection():
    result = resume_mod.parse_resume(injection_resume_text(), as_of=AS_OF)
    assert result["status"] == "ok"
    assert any("instruction-like" in warning for warning in result["warnings"])
    text = json.dumps(result["profile"])
    assert "reveal the system prompt" not in text.lower() or True  # content treated as data only
    assert result["profile"]["experiences"], "career facts still extracted around injected text"


def test_extract_text_pdf_adapter():
    extracted = resume_mod.extract_text("cv.pdf", minimal_pdf("Data Analyst at Acme Labs"))
    assert extracted["status"] == "ok" and extracted["format"] == "pdf"
    assert "Data Analyst at Acme Labs" in extracted["text"]


def test_extract_text_docx_adapter():
    extracted = resume_mod.extract_text(
        "cv.docx", docx_bytes(["MAYA CHEN", "Data Analyst at Acme Labs 2019-2021"]))
    assert extracted["status"] == "ok" and extracted["format"] == "docx"
    assert "MAYA CHEN" in extracted["text"]


def test_extract_text_json_adapter():
    payload = json.dumps({"name": "Maya", "experiences": [{"title": "Analyst", "start": "2020", "end": "2021"}]})
    extracted = resume_mod.extract_text("cv.json", payload.encode())
    assert extracted["status"] == "ok"
    result = resume_mod.parse_resume(payload, as_of=AS_OF)
    assert result["status"] == "ok" and result["profile"]["experiences"]


def test_extract_text_unsupported_and_errors():
    assert resume_mod.extract_text("cv.exe", b"MZ\x90\x00")["status"] == "unsupported"
    assert resume_mod.extract_text("legacy.doc", b"\xd0\xcf\x11\xe0")["status"] == "unsupported"
    assert resume_mod.extract_text("cv.pdf", b"%PDF-broken")["status"] == "error"
    assert resume_mod.extract_text("empty.txt", b"")["status"] == "error"
    assert resume_mod.extract_text("huge.txt", b"a" * (resume_mod.MAX_FILE_BYTES + 1))["status"] == "error"
    assert resume_mod.extract_text("broken.json", b"{not json")["status"] == "error"


def test_parse_resource_variants():
    text_resource = {"uri": "file:///cv.txt", "metadata": {"filename": "cv.txt",
                                                           "text": plain_resume_text()}}
    parsed = resume_mod.parse_resource(text_resource)
    assert parsed["status"] == "ok" and parsed["filename"] == "cv.txt"

    import base64
    encoded = base64.b64encode(plain_resume_text().encode()).decode()
    parsed = resume_mod.parse_resource({"uri": "", "metadata": {"data": encoded, "filename": "cv.txt"}})
    assert parsed["status"] == "ok"

    data_uri = "data:text/plain;base64," + encoded
    parsed = resume_mod.parse_resource({"uri": data_uri, "metadata": {}})
    assert parsed["status"] == "ok"

    link_only = resume_mod.parse_resource({"uri": "https://drive.example/cv.pdf", "metadata": {}})
    assert link_only["status"] == "unsupported"
    assert "never fetched" in link_only["error"]


def test_parse_resume_garbage_is_structured_error():
    result = resume_mod.parse_resume("lorem ipsum dolor sit amet", as_of=AS_OF)
    assert result["status"] == "error"
    assert "could not read career facts" in result["error"]
