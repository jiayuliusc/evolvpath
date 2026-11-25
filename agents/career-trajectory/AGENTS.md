# AGENTS.md

You are building a **uAgent** that people can talk to in **ASI:One**. Its brain is a **LangChain agent loop** powered by the **ASI:One chat completions API**. It speaks the **Chat Protocol**, replies with **interactive cards**, and is **registered on Agentverse** through a mailbox.

The user's prompt only says *what* the agent does (its domain, its tools, its data source). Everything about *how* to build, run and register it is in this file. Follow it top to bottom. Don't ask the user anything this file already answers.

---

## 1. Definition of done

You're finished only when all of these are true:

1. `uv run ruff check .` and `uv run pytest -q` pass.
2. The agent runs locally (`uv run python agent.py`) and logs its address (`agent1q…`).
3. The agent is connected to Agentverse as a **mailbox** agent (section 7), with a name, short description and README.
4. The Agentverse MCP (or the fallback in section 7) confirms that the agent exists on your Agentverse account.
5. You've given the user:
   - the agent address
   - the Agentverse profile link `https://agentverse.ai/agents/details/<address>/profile`
   - two example prompts to try in ASI:One

## 2. Ground rules

- **Secrets live only in `.env`.** Never print, log, echo or commit an API key or the seed. Never paste a key into code, the README or chat. Read keys with `python-dotenv`.
- If a required key is missing from `.env`, follow section 3. **Never invent or reuse a placeholder key.**
- Use `uv` for everything (`uv init`, `uv add`, `uv run`). Python 3.11–3.13.
- Keep the agent small: the 5 files in section 4, plus tests.
- No database. Use `ctx.storage` (the agent's built-in key-value store) for any per-user state, keyed by `sender`.
- Don't add payments.
- Prefer free, keyless public APIs for the agent's data source unless the user says otherwise. Always set a timeout of 10s or less on every HTTP call.

## 3. API keys and `.env`

The project root needs a `.env` with:

```dotenv
ASI_ONE_API_KEY=        # ASI:One chat completions
AGENTVERSE_API_KEY=     # Agentverse registration + MCP
AGENT_SEED=             # secret seed that fixes the agent's address
ASI_ONE_MODEL=asi1      # optional; asi1 | asi1-mini | asi1-ultra
```

Create `.env.example` with the same keys and empty values, and add `.env` to `.gitignore`.

**If a key is missing or empty:**

- **`AGENT_SEED`:** generate it yourself and write it to `.env` without showing it:
  `uv run python -c "import secrets;print(secrets.token_hex(32))"`
  Keep it forever. A new seed means a new agent address.
- **`ASI_ONE_API_KEY`:** stop and ask the user to create one, then wait. Tell them:
  1. Go to https://asi1.ai and sign in.
  2. Open the **Developer** section.
  3. Click **Create New**, name the key, and copy it.
  4. Paste it into `.env` as `ASI_ONE_API_KEY=...`.
- **`AGENTVERSE_API_KEY`:** stop and ask the user to create one, then wait. Tell them:
  1. Go to https://agentverse.ai and sign in.
  2. Open **Profile → API Keys** (https://agentverse.ai/profile/api-keys) and create a key.
  3. Paste it into `.env` as `AGENTVERSE_API_KEY=...`.

After the user says they've added a key, check it's present without printing it:
`uv run python -c "import os,dotenv;dotenv.load_dotenv();print({k: bool(os.getenv(k)) for k in ['ASI_ONE_API_KEY','AGENTVERSE_API_KEY','AGENT_SEED']})"`

## 4. Project layout

```
.
├── AGENTS.md         # this file
├── CLAUDE.md         # points here
├── .env              # secrets (never commit)
├── .env.example
├── .gitignore
├── pyproject.toml
├── agent.py          # uAgent: identity, Chat Protocol handlers, mailbox
├── brain.py          # LangChain agent loop + domain tools
├── cards.py          # interactive card builders + selection parsing
├── README.md         # Agentverse profile README (people and ASI:One read it)
└── tests/
    └── test_brain_and_cards.py
```

Setup:

```bash
uv init --python 3.12 --no-workspace .   # skip if pyproject.toml exists
uv add "uagents>=0.25" "uagents-core>=0.4.9" "langchain>=1.0" "langchain-openai>=1.0" httpx python-dotenv
uv add --dev pytest pytest-asyncio ruff
```

## 5. The brain: a LangChain agent loop on ASI:One (`brain.py`)

ASI:One's API is OpenAI-compatible:
- base URL `https://api.asi1.ai/v1`
- `Authorization: Bearer $ASI_ONE_API_KEY`
- supports tool calling

So use `ChatOpenAI` with a different `base_url`, and LangChain v1's `create_agent`. That gives you the ReAct loop: the model decides, a tool runs, the result goes back to the model, and it repeats until the model answers without a tool call.

```python
import json, os
import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    base_url="https://api.asi1.ai/v1",
    api_key=os.environ["ASI_ONE_API_KEY"],
    model=os.getenv("ASI_ONE_MODEL", "asi1"),
    temperature=0,
    timeout=30,
)

@tool
def lookup(query: str) -> str:
    """One clear sentence: what this tool does and what input it expects."""
    with httpx.Client(timeout=10) as http:
        data = http.get("https://...", params={...}).json()
    # Return JSON with a title and FLAT facts; cards.py turns these into a card.
    return json.dumps({"title": "...", "facts": {"Label": "value", ...}})

SYSTEM_PROMPT = """You are <agent name>. <One line: what you do and for whom.>
Always call a tool to get facts; never make them up.
If a required input is missing, ask ONE short question and call no tools.
Answer in 1-3 short sentences."""

agent = create_agent(llm, tools=[lookup], system_prompt=SYSTEM_PROMPT)

async def answer(text: str) -> tuple[str, dict | None]:
    """Run the loop. Return (reply_text, last_tool_result_or_None)."""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": text}]})
    msgs = result["messages"]
    tool_data = None
    for m in reversed(msgs):
        if isinstance(m, ToolMessage):
            try:
                tool_data = json.loads(m.content)
            except (TypeError, ValueError):
                pass
            break
    return msgs[-1].content, tool_data
```

Rules for the brain:
- **One tool per real capability.** Each tool's docstring is what the model reads to decide when to call it, so make it precise: purpose, input format, units.
- If the data source needs a lookup step first (e.g. a name to coordinates), make that a second tool or do it inside the main tool. Never ask the model to guess IDs or coordinates.
- **Tools return JSON strings**, never raise. On failure, return `{"error": "<short reason>"}` so the model can explain it.
- Keep the system prompt short. Behavior belongs in tools and in this structure, not in long prompts.
- The loop must end: `create_agent` stops when the model answers without a tool call. Don't add retries that could loop.

## 6. The Chat Protocol and cards (`agent.py`, `cards.py`)

### 6.1 Chat Protocol (required for ASI:One)

ASI:One talks to agents only through the **Chat Protocol** (`uagents_core.contrib.protocols.chat`):

- A `ChatMessage` (`timestamp`, `msg_id`, `content: list[...]`) is answered with a `ChatAcknowledgement` (`acknowledged_msg_id`).
- Content types:
  - `TextContent`
  - `ResourceContent` (files, images)
  - `MetadataContent` (cards; `dict[str, str]`)
  - `StartSessionContent` / `EndSessionContent`
  - `StartStreamContent` / `EndStreamContent`

```python
import os
from datetime import datetime, timezone
from uuid import uuid4
from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, EndSessionContent, TextContent, chat_protocol_spec,
)
import brain, cards

load_dotenv()

agent = Agent(
    name="<agent-name>",                 # short, descriptive, unique
    seed=os.environ["AGENT_SEED"],
    port=8000,
    mailbox=True,                        # reachable via Agentverse, even behind Wi-Fi
    description="<one line: what it does, for whom>",
    readme_path="README.md",
    publish_agent_details=True,
)
chat = Protocol(spec=chat_protocol_spec)

@chat.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id))
    text = cards.normalize_incoming(msg.text())     # strips @mentions, decodes card selections
    if not text:
        return
    try:
        reply, data = await brain.answer(text)
    except Exception:
        ctx.logger.exception("brain failed")
        reply, data = "Sorry, something went wrong. Please try again.", None
    await ctx.send(sender, cards.build_reply(reply, data))

@chat.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, ack: ChatAcknowledgement):
    pass

agent.include(chat, publish_manifest=True)   # publishes the protocol manifest: required for ASI:One

if __name__ == "__main__":
    agent.run()
```

Rules:
- **Acknowledge first**, then do the work.
- Always handle `ChatAcknowledgement`, even with a no-op.
- `publish_manifest=True` is mandatory. Without it ASI:One won't route to the agent.
- `sender` is a verified address. Key any per-user state by it: `ctx.storage.set(f"prefs:{sender}", ...)`.
- Strip a leading `@handle ` mention from incoming text before using it.

### 6.2 Interactive cards (`cards.py`)

ASI:One renders a card when a `ChatMessage` contains **one `TextContent` (the narration and fallback) plus one `MetadataContent`** with these keys. Every value must be a **string**:

| Key | Value |
| --- | --- |
| `card_protocol_version` | `"1"` (required; without it the card is silently dropped) |
| `requires_card_interaction` | `"true"` |
| `card_kind` | `"detail"`, `"form"`, `"carousel"`, `"review"` or `"custom"` |
| `card_payload` | `json.dumps(payload)`, at the top level (no wrapper object), at most 64 KB |
| `is_terminal` (optional) | `"true"` for informational cards with no input. **Never** on a card that needs input, or the drawer won't open. |
| `preferred_drawer_width_px` (optional) | e.g. `"540"` (clamped 320–800) |

If the card fails validation, ASI:One shows only the text, so the text must stand on its own.

Payload shapes:

```python
# detail: show a result
{"title": "...", "summary_rows": [{"label": "...", "value": "..."}],
 "ctas": [{"label": "Check another", "selection": {"action": "new_query"}, "primary": True}]}

# form: ask for missing input
{"title": "...", "fields": [{"name": "query", "kind": "text", "label": "...", "required": True}],
 "submit_cta": {"label": "Submit", "selection": {"action": "submit"}}}
# field kinds: text | number | email | select (needs "options": [{"value","label"}]) | checkbox

# carousel: pick one of several
{"title": "...", "items": [{"id": "...", "title": "...", "subtitle": "...", "secondary_text": "...",
  "primary_cta": {"label": "Select", "selection": {"action": "pick", "id": "..."}}}]}

# review: confirm or cancel
{"title": "...", "summary_rows": [...],
 "approve_cta": {"label": "Confirm", "selection": {"action": "confirm"}, "primary": True},
 "reject_cta": {"label": "Cancel", "selection": {"action": "cancel"}}}
```

What `cards.py` must provide:
- `build_reply(reply, data)`:
  - If `data` has `facts`: return text plus a **`detail`** card (`title` and one `summary_rows` entry per fact, all values as strings).
  - If `data` is `None` (the model asked a clarifying question): return text plus a **`form`** card with one required text field for the missing input.
  - If `data` has `error`: return text only.
  - Finish every reply with `EndSessionContent()` only when nothing more is expected.
- `normalize_incoming(text)`. Card taps come back as a `TextContent`, in one of two forms:
  - **JSON** (direct `@mention`), e.g. `{"query": "...", "action": "submit"}`. Try `json.loads` first and turn it into a plain sentence for the brain (e.g. `"query: Paris"`). Treat `{"action": "new_query"}` as "ask what to look up next".
  - **Prose** (routed through ASI:One's planner, e.g. "The user submitted query Paris"). If JSON parsing fails, pass the text through unchanged.
- Check payloads in the playground: https://asi1.ai/developer/card-playground

## 7. Register on Agentverse

Registration is a **mailbox** connection. The running agent proves it owns its address by signing an Agentverse challenge with its seed, so the connect step must go through the running agent.

### 7.1 Write the profile README (`README.md`)

ASI:One reads this README to decide whether to call the agent. Write it for that reader:

```markdown
![tag:chatprotocol](https://img.shields.io/badge/chatprotocol-1D3BD4)

# <Agent name>
<One sentence: what it does and for whom.>

## What it can do
- <capability>
- <capability>

## Example prompts
- <a real thing a user would type>
- <another>

## Data source
<Name of the public API it uses.>
```

No secrets, no internal URLs.

### 7.2 Run and connect

```bash
uv run python agent.py        # run in the background; keep it running
```

Read the agent address from the log. Then connect the mailbox using the agent's own `/connect` endpoint, which is what the Agent Inspector's **Connect** button does:

```bash
set -a; source .env; set +a
curl -s -X POST http://127.0.0.1:8000/connect \
  -H "Content-Type: application/json" \
  -d "{\"user_token\": \"$AGENTVERSE_API_KEY\", \"agent_type\": \"mailbox\"}"
# expect {"success": true}
```

The agent then publishes its name, description and README to Agentverse, because `publish_agent_details=True`.

**Fallback:** open the Agent Inspector link printed in the agent's log, click **Connect → Mailbox**, and sign in to Agentverse.

### 7.3 Verify and polish with the Agentverse MCP

The Agentverse MCP server lets you manage Agentverse directly from your coding assistant. If its tools (names like `list_user_agents_mailbox`, `get_user_agent`, `search_agents`) aren't available in this session, tell the user how to add it, then continue with the browser checks below. **Never block on the MCP.**

Add it to Claude Code (the user runs this; the key comes from `.env`):

```bash
claude mcp add --transport sse agentverse https://mcp.agentverse.ai/sse \
  --header "Authorization: Bearer $AGENTVERSE_API_KEY"
# lighter alternative: --transport http agentverse-lite https://mcp-lite.agentverse.ai/mcp
```

Use these tools in this order. Tools that take `api_token` get the value of `AGENTVERSE_API_KEY`, which you read from `.env` and never print:

1. `list_user_agents_mailbox` or `get_user_agent(address)`: confirm the agent exists.
2. `update_user_agent_mailbox(address, name, short_description, readme)`: if the name, description or README is missing or stale, push them from `README.md`.
3. `search_agents(search_text="<what a user would ask>")`: check whether it's findable. New agents can take a few minutes to be indexed.
4. Optional:
   - `start_agent_seo_eval(address)`, then `get_agent_seo_eval_results(address)`: get a discoverability score and suggestions.
   - `get_latest_logs_for_user_agent`: only for hosted agents. This agent is local, so read its own terminal log.

**Without MCP:** open `https://agentverse.ai/agents/details/<address>/profile` and check that the name, README and Chat Protocol are shown.

### 7.4 Test end to end

- On the agent's Agentverse page, use **Chat with Agent** (or the **Testing** tab) and send an example prompt. You should see the reply text and a card.
- In https://asi1.ai, ask a question matching an example prompt, or mention the agent directly.
- The local agent must keep running for ASI:One to reach it.

## 8. Tests (`tests/test_brain_and_cards.py`)

- Tools: mock HTTP (monkeypatch `httpx.Client.get`). Assert each tool returns valid JSON with `title` and `facts`, or with `error` on failure.
- Cards: `build_reply` gives one `TextContent` plus one `MetadataContent` whose values are all strings. `card_payload` must parse with `json.loads`, and `card_protocol_version == "1"`.
- `normalize_incoming`: handles selection JSON, prose, and `@handle` prefixes.
- Never call the real ASI:One API or Agentverse in tests.

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `/connect` returns 401 | `AGENTVERSE_API_KEY` is wrong or expired. Ask the user for a new one (section 3). |
| Port 8000 in use | Change `port=` in `Agent(...)` and in the `curl` URL |
| ASI:One doesn't route to the agent | Check `publish_manifest=True`, that the agent is running, that the mailbox is connected, and that the README has example prompts. Wait a few minutes for indexing. |
| Card doesn't render | Some value isn't a string, `card_protocol_version` is missing, the payload is wrapped, or it's over 64 KB. Validate in the card playground. |
| Model never calls the tool | Make the tool docstring more specific, keep `temperature=0`, and try `ASI_ONE_MODEL=asi1`. |
| `KeyError: ASI_ONE_API_KEY` | `.env` is missing or not loaded. See section 3. |
