"""Interactive-card builders and incoming-message normalization for the Chat Protocol.

Every card is one MetadataContent whose metadata values are ALL strings, with
card_protocol_version "1" and an unwrapped card_payload under 64 KB, so ASI:One
can render it; the narration TextContent always stands on its own.
"""

import json
import re
from datetime import UTC, datetime
from uuid import uuid4

from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    EndSessionContent,
    MetadataContent,
    TextContent,
)

CARD_PROTOCOL_VERSION = "1"
PLAN_GOALS = "plan my goals"  # sentinel sentence that triggers the agent-3 handoff
_NEW_QUERY = "start a new trajectory"
_MAX_PAYLOAD_BYTES = 65536  # 64 KB card_payload cap

_FIELD_LABELS = {
    "target_role": "target role",
    "query": "target role",
    "history": "career history",
    "value": "answer",
    "text": "answer",
}


def normalize_incoming(text: str) -> str:
    """Normalize an incoming message: strip @mentions and decode card-tap selections
    into plain sentences for the brain. Prose passes through unchanged."""
    stripped = re.sub(r"^@[\w.-]+\s+", "", (text or "").strip())
    try:
        data = json.loads(stripped)
    except ValueError:
        return stripped  # prose (including planner-routed taps) passes through
    if not isinstance(data, dict):
        return stripped
    action = str(data.get("action") or "").lower()
    if not action:
        return stripped  # user-pasted JSON such as an alumni_history — keep verbatim
    if action == "plan_goals":
        return PLAN_GOALS
    if action == "new_query":
        return _NEW_QUERY
    if action == "pick":
        return f"select path {data.get('id', '')}".strip()
    # submit (and any other action): flatten remaining keys into a plain sentence
    parts = []
    for key, value in data.items():
        if key == "action" or value is None:
            continue
        label = _FIELD_LABELS.get(str(key), str(key).replace("_", " "))
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        parts.append(f"{label}: {value}")
    return "; ".join(parts) or "submit"


def _chat_message(content: list) -> ChatMessage:
    return ChatMessage(timestamp=datetime.now(UTC), msg_id=str(uuid4()), content=content)


def _metadata(kind: str, payload: dict, terminal: bool = False) -> MetadataContent:
    dumped = json.dumps(payload)
    if len(dumped) > _MAX_PAYLOAD_BYTES:  # keep the card renderable; text still stands alone
        payload = {
            "title": str(payload.get("title", ""))[:120],
            "summary_rows": [{"label": "Result", "value": dumped[:8000]}],
        }
        dumped = json.dumps(payload)
    metadata = {
        "card_protocol_version": CARD_PROTOCOL_VERSION,
        "requires_card_interaction": "true",
        "card_kind": kind,
        "card_payload": dumped,
    }
    if terminal:
        metadata["is_terminal"] = "true"
    return MetadataContent(metadata=metadata)


def _rows(facts: dict, cap: int = 12) -> list[dict]:
    return [
        {"label": str(k)[:60], "value": str(v)[:400]}
        for k, v in list(facts.items())[:cap]
        if v is not None
    ]


def _detail_card(title: str, facts: dict) -> dict:
    payload = {"title": str(title)[:120], "summary_rows": _rows(facts)}
    return {"meta": _metadata("detail", payload, terminal=True), "terminal": True}


def _paths_card(data: dict) -> dict:
    recommended = str(data.get("recommended_id") or "")
    items = []
    for path in data["paths"][:3]:
        path_id = str(path.get("id") or "")
        gaps = [str(g) for g in (path.get("skill_gaps") or [])][:5]
        subtitle = str(path.get("summary") or "")
        if path_id and path_id == recommended:
            subtitle = f"Recommended. {subtitle}"[:200]
        items.append(
            {
                "id": path_id,
                "title": str(path.get("title") or path_id)[:120],
                "subtitle": subtitle,
                "secondary_text": ("Skills to build: " + "; ".join(gaps))[:200] if gaps else "",
                "primary_cta": {"label": "Select", "selection": {"action": "pick", "id": path_id}},
            }
        )
    payload = {"title": str(data.get("title") or "Career path options")[:120], "items": items}
    return {"meta": _metadata("carousel", payload), "terminal": False}


def _trajectory_card(data: dict) -> dict:
    traj = data["trajectory"]
    target = str(data.get("target_role") or "")
    scores = data.get("scores") or {}
    rows = []
    for phase in traj.get("phases") or []:
        label = f"Phase {phase.get('n', '')}".strip()
        if phase.get("horizon"):
            label = f"{label} ({phase['horizon']})"
        rows.append({"label": label[:60], "value": str(phase.get("goal") or "")[:400]})
    if traj.get("skill_gaps"):
        rows.append(
            {"label": "Skill gaps", "value": "; ".join(str(g) for g in traj["skill_gaps"])[:400]}
        )
    if traj.get("roles"):
        rows.append({"label": "Role sequence", "value": " -> ".join(str(r) for r in traj["roles"])[:400]})
    if traj.get("estimated_months"):
        rows.append({"label": "Estimated horizon", "value": f"about {traj['estimated_months']} months (guidance only)"})
    if scores.get("readiness") is not None:
        rows.append({
            "label": "Readiness / transferability",
            "value": f"{scores.get('readiness')}/100 - {scores.get('transferability')}/100 "
                     f"(difficulty: {scores.get('transition_difficulty')})"[:400],
        })
    if traj.get("confidence") is not None:
        rows.append({"label": "Confidence", "value": f"{traj['confidence']:.0%}"})
    if traj.get("risks"):
        rows.append({"label": "Risks", "value": "; ".join(str(r) for r in traj["risks"][:4])[:400]})
    if target:
        rows.append({"label": "Target role", "value": target[:400]})
    payload = {
        "title": str(traj.get("title") or f"Trajectory toward {target or 'your goal'}")[:120],
        "summary_rows": rows,
        "ctas": [
            {"label": "Plan my goals", "selection": {"action": "plan_goals"}, "primary": True},
            {"label": "New trajectory", "selection": {"action": "new_query"}},
        ],
    }
    return {"meta": _metadata("detail", payload), "terminal": False}


def _form_card(needs_input: dict) -> dict:
    field = str(needs_input.get("field") or "target_role")
    label = str(needs_input.get("label") or "What target role should the trajectory aim for?")
    payload = {
        "title": str(needs_input.get("title") or "One quick question")[:120],
        "fields": [{"name": field, "kind": "text", "label": label[:200], "required": True}],
        "submit_cta": {"label": "Submit", "selection": {"action": "submit"}},
    }
    return {"meta": _metadata("form", payload), "terminal": False}


def build_reply(reply: str, data: dict | None) -> ChatMessage:
    """Build the outgoing ChatMessage: narration text plus one card when data allows.

    - facts -> detail card (informational, terminal)
    - paths -> carousel card (pick one)
    - trajectory -> detail card with Plan-my-goals CTA
    - needs_input -> form card with one required text field
    - error or no data -> text only
    EndSessionContent only when nothing more is expected from the user.
    """
    reply = (reply or "").strip() or "Here's what I found."
    data = data if isinstance(data, dict) else None
    content: list = [TextContent(text=reply)]
    card = None
    if data:
        if isinstance(data.get("error"), str):
            pass  # error -> text only
        elif isinstance(data.get("trajectory"), dict):
            card = _trajectory_card(data)
        elif isinstance(data.get("paths"), list) and data["paths"]:
            card = _paths_card(data)
        elif isinstance(data.get("facts"), dict) and data["facts"]:
            card = _detail_card(str(data.get("title") or "Alumni profile"), data["facts"])
        elif isinstance(data.get("needs_input"), dict):
            card = _form_card(data["needs_input"])
    if card:
        content.append(card["meta"])
        if card["terminal"]:
            content.append(EndSessionContent())
    elif data is None or isinstance(data.get("error"), str):
        content.append(EndSessionContent())
    return _chat_message(content)
