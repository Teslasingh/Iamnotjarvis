# Iamnotjarvis

Local web agent powered by Azure OpenAI. It can use host tools for shell commands and filesystem operations with the same permissions as the running Python process.

## Setup

Copy `.env.example` to `.env` and fill:

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
DEPLOYMENT_NAME=
OPENAI_API_VERSION=2024-12-01-preview
FRIDAY_PASSWORD=change-me
SESSION_SECRET=change-this-long-random-string
PORT=80
HOST=0.0.0.0
ALLOW_SHELL=true
MAX_AGENT_STEPS=32
LLM_TIMEOUT_SECONDS=90
LLM_RETRIES=3
EVENT_RING_MAX=500
```

## Run

Python 3.8+ required.

```bash
cd ~/Documents/Iamnotjarvis
python3 -m pip install --user -e .
python3 -m friday
```

Open `http://127.0.0.1` or the configured host/port.

## Notes

- `.env` is gitignored. Do not commit secrets.
- `FRIDAY_PASSWORD` protects the page/API/WebSocket with a signed cookie.
- The agent has broad shell/filesystem access. Do not expose it to untrusted networks.
- Conversation memory is in RAM only and limited by `MEMORY_RECENT_TURNS`; it is cleared on server start, logout, and shutdown.
- Long-term **soul memory** is stored in `soul.md` at the agent workdir root (`AGENT_WORKDIR`, or cwd when unset). Friday recalls it every turn and selectively updates it after chats (preferences, behaviors, key learnings — not full transcripts). You can edit `soul.md` directly or reset it with `DELETE /api/soul`. Toggle with `SOUL_ENABLED` / `SOUL_AUTO_UPDATE`.
- Set `AGENT_WORKDIR` to the repo root (e.g. `~/Documents/Iamnotjarvis`) so `soul.md` lives alongside the project. Friday can edit its own `friday/` source via file tools when soul or user requests it; **restart** `python -m friday` after Python source changes (not hot-reloaded).
- Optional `QUERY_EXPANSION_ENABLED=true` rewrites vague prompts with the LLM before each agent turn (see `.env.example`). This now runs as part of **task analysis**, which also captures intent, edge cases, and complexity routing.
- **Multi-agent orchestration** (`MULTI_AGENT_ENABLED=true`) deploys sequential explore → execute → verify sub-agents only when task analysis determines the request is complex enough to warrant it. Simple tasks stay on the single-agent path. Set `MULTI_AGENT_ENABLED=false` to disable orchestration while keeping analysis.
- On Jetson Nano, use Python 3.8+ (`python3.8 -m friday` if needed).
