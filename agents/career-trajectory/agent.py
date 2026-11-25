"""uAgent entry point: identity, Chat Protocol handlers, mailbox connection.

Stage-2 additions: résumé files arriving as ResourceContent are parsed
deterministically (raw file text never enters the LLM prompt), session state
now carries the canonical profile / benchmark / analysis / plan / selection,
and the Goal Planning handoff emits a validated schema-2.0 payload.

Privacy: logs carry trace ids and counts only - never raw résumé content or
personal contact details (see career_profile.redact / summarize_for_prompt).
"""

import json
import os
import uuid
from datetime import UTC, datetime

from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    ResourceContent,
    TextContent,
    chat_protocol_spec,
)

import brain
import cards
import career_profile as pf
import resume as resume_mod
import trajectories

load_dotenv()

AGENT3_ADDRESS = os.getenv("AGENT_3_ADDRESS", "").strip()

agent = Agent(
    name="PathFinder",
    seed=os.environ["AGENT_SEED"],
    port=8000,
    mailbox=True,  # reachable via Agentverse even behind NAT
    description=(
        "Turns LinkedIn histories and résumés into evidence-based career trajectories "
        "with readiness scores, gap analysis and path options, ready for goal planning."
    ),
    readme_path="README.md",
    publish_agent_details=True,
)
chat = Protocol(spec=chat_protocol_spec)

_STATE_KEY = "state:{sender}"
# Per-session engine state, all keyed by the verified sender address.
_CONTEXT_KEYS = ("profile", "benchmark", "analysis", "plan", "selection", "trajectory", "target_role")
_CONTEXT_CHARS = 9000  # cap on the context block appended to the prompt


def _now() -> datetime:
    return datetime.now(UTC)


def _trace(ctx: Context) -> str:
    return f"trace_{uuid.uuid4().hex[:8]}"


def _resources(msg: ChatMessage) -> list[dict]:
    """Extract Chat Protocol resources (files) from a message."""
    found = []
    for item in msg.content or []:
        if not isinstance(item, ResourceContent):
            continue
        resource = item.resource if isinstance(item.resource, dict) else \
            (item.resource[0] if isinstance(item.resource, list) and item.resource else None)
        if resource:
            found.append({"uri": resource.uri, "metadata": dict(resource.metadata or {})})
    return found


def _attach_files(ctx: Context, sender: str, state: dict, resources: list[dict], trace: str) -> tuple[str, str]:
    """Parse résumé resources deterministically; return (note for the LLM, fragment JSON).

    Raw file bytes/text never reach the model: only the extracted, sanitized
    profile fragment is stored and (compactly) attached to the prompt context.
    """
    notes: list[str] = []
    stored = state.get("resume_fragment")
    for resource in resources:
        parsed = resume_mod.parse_resource(resource)
        filename = pf.redact(parsed.get("filename") or "resume")
        if parsed.get("status") != "ok":
            reason = pf.redact(parsed.get("error") or "could not parse the file")
            ctx.logger.info(f"[{trace}] resume parse failed: status={parsed.get('status')} file={filename}")
            notes.append(f"[Résumé file '{filename}' could not be used: {reason}]")
            continue
        fragment = parsed["profile"]
        warnings = parsed.get("warnings") or []
        if state.get("profile"):
            fragment = pf.reconcile(state["profile"], fragment)
            state["profile"] = fragment
        else:
            state["resume_fragment"] = fragment
        stored = fragment
        ctx.logger.info(
            f"[{trace}] resume parsed: file={filename} chars_in={len(str(resource.get('metadata') or ''))} "
            f"roles={len(fragment.get('experiences') or [])} skills={len(fragment.get('skills') or [])} "
            f"warnings={len(warnings)}"
        )
        notes.append(f"[Résumé file '{filename}' was parsed: {len(fragment.get('experiences') or [])} roles, "
                     f"{len(fragment.get('skills') or [])} skills, {len(warnings)} parsing warning(s). "
                     f"Use analyze_readiness with the stored resume fragment JSON from context.]")
    return " ".join(notes), json.dumps(stored) if stored else ""


def _context_block(state: dict, resume_json: str) -> str:
    """Compact session context for the model. Bounded and sanitized; carries the
    JSON the tools need (profile, benchmark, analysis, plan, selection)."""
    parts = []
    if state.get("profile"):
        parts.append('"profile": ' + json.dumps(pf.summarize_for_prompt(state["profile"])))
    if resume_json:
        parts.append('"resume_fragment": ' + resume_json)
    if state.get("benchmark"):
        parts.append('"benchmark": ' + json.dumps(state["benchmark"]))
    if state.get("analysis"):
        parts.append('"analysis": ' + json.dumps(state["analysis"]))
    if state.get("plan"):
        parts.append('"plan": ' + json.dumps(state["plan"]))
    if state.get("selection"):
        parts.append('"selection": ' + json.dumps(state["selection"]))
    if state.get("target_role"):
        parts.append('"target_role": ' + json.dumps(state["target_role"]))
    blob = "{" + ", ".join(parts) + "}"
    if len(blob) > _CONTEXT_CHARS:  # keep the prompt bounded; newest state wins
        blob = blob[:_CONTEXT_CHARS] + '..."}'
    return blob


async def _hand_off(ctx: Context, sender: str, state: dict) -> None:
    """Emit the validated schema-2.0 payload as copy-ready JSON; forward to
    agent 3 when AGENT_3_ADDRESS is set. Forwarding failures degrade gracefully."""
    trace = _trace(ctx)
    profile, analysis, selection = state.get("profile"), state.get("analysis"), state.get("selection")
    if not (profile and analysis and selection and selection.get("trajectory")):
        await ctx.send(
            sender,
            cards.build_reply(
                "I don't have a selected trajectory yet - share your history, get your analysis, "
                "pick a path, and then I'll hand it to goal planning.",
                None,
            ),
        )
        return
    payload = trajectories.build_payload(profile, analysis, selection)
    errors = trajectories.validate_payload(payload)
    if errors:  # never forward an invalid payload
        ctx.logger.error(f"[{trace}] invalid trajectory payload: {errors[:3]}")
        await ctx.send(
            sender,
            cards.build_reply(
                "The trajectory payload failed validation, so I did not forward it: "
                + "; ".join(errors[:3]) + ". Ask me to regenerate the analysis.",
                {"error": "invalid trajectory payload"},
            ),
        )
        return
    body = trajectories.dumps(payload)
    if AGENT3_ADDRESS:
        try:
            await ctx.send(
                AGENT3_ADDRESS,
                ChatMessage(
                    timestamp=_now(),
                    msg_id=str(uuid.uuid4()),
                    content=[TextContent(text=f"Trajectory for goal planning: {body}")],
                ),
            )
            ctx.logger.info(f"[{trace}] trajectory {payload['trajectory_id']} forwarded to goal planning")
            note = " It's on its way to the Goal Planning agent now."
        except Exception:
            ctx.logger.exception(f"[{trace}] forwarding trajectory to the Goal Planning agent failed")
            note = " Direct forwarding failed - please copy the JSON to the Goal Planning agent."
    else:
        note = " Copy it to the Goal Planning agent, or set AGENT_3_ADDRESS to auto-forward."
    await ctx.send(
        sender,
        cards.build_reply(f"Here's the goal-planning payload:\n\n{body}\n{note}", None),
    )


def _merge_state(state: dict, data: dict) -> dict:
    """Fold a tool result into per-user session state.

    choose_path returns its selection flattened at the top level (title, selected,
    scores, trajectory, alternatives); rebuild the nested "selection" object the
    handoff and prompt context expect, so "Plan my goals" works after a card tap.
    """
    for source in _CONTEXT_KEYS:
        if isinstance(data.get(source), (dict, str)) and data.get(source):
            state[source] = data[source]
    if data.get("trajectory") and data.get("selected"):
        state["selection"] = {
            "title": data.get("title") or "",
            "target_role": data.get("target_role"),
            "selected": data.get("selected"),
            "type": data.get("type"),
            "scores": data.get("scores") or {},
            "trajectory": data.get("trajectory"),
            "alternatives": data.get("alternatives") or [],
        }
    if isinstance(data.get("profile"), dict):
        state.pop("resume_fragment", None)  # merged into the unified profile
    return state


@chat.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage) -> None:
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=_now(), acknowledged_msg_id=msg.msg_id),
    )
    trace = _trace(ctx)
    text = cards.normalize_incoming(msg.text())  # strips @mentions, decodes card selections
    resources = _resources(msg)

    key = _STATE_KEY.format(sender=sender)
    state = ctx.storage.get(key) or {}

    if not text and not resources:
        return
    if text == cards.PLAN_GOALS:
        await _hand_off(ctx, sender, state)
        return

    file_note, resume_json = _attach_files(ctx, sender, state, resources, trace)

    prompt = text or "I attached a résumé file - analyze it."
    if file_note:
        prompt = f"{file_note}\n\n{prompt}"
    context = _context_block(state, resume_json)
    if context and context.strip("{}"):
        prompt = f"{prompt}\n\n[Session context (stored JSON; pass these into tools as needed): {context}]"

    try:
        reply, data = await brain.answer(prompt)
    except Exception:
        ctx.logger.exception(f"[{trace}] brain failed")
        reply, data = "Sorry, something went wrong. Please try again.", None

    if isinstance(data, dict):
        ctx.storage.set(key, _merge_state(state, data))

    await ctx.send(sender, cards.build_reply(reply, data))


@chat.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, ack: ChatAcknowledgement) -> None:
    pass


agent.include(chat, publish_manifest=True)  # required for ASI:One routing

if __name__ == "__main__":
    agent.run()
