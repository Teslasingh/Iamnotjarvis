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
cd ~/Documents/Iamnotjarvis/friday
python3 -m pip install --user -e .
python3 -m friday
```

Open `http://127.0.0.1` or the configured host/port.

## Terminal / Shell Support

✅ **Status:** Shell execution is **enabled** when `ALLOW_SHELL=true` (default).


**Enabled by default** and gated by environment flags. Commands run **non-interactively** with captured stdout/stderr and exit codes, under the same OS permissions as the Python process.

### Key flags
- `ALLOW_SHELL=true|false` — master on/off switch.
- `FRIDAY_WINDOWS_SHELL=powershell|cmd` — Windows shell selector (default: `powershell`).
- `FRIDAY_UNIX_LOGIN_SHELL=true|false` — use `bash --login` on Unix to load profile PATH.

### Execution modes
- **Finite commands:** `run_shell` (must exit; stdout/stderr captured; exit code reported).
- **Long-running apps:** `start_shell_job` (watchdog-enforced limits; stoppable).

### Safety & limits
- **Watchdog** stops runaway jobs (max runtime, stalled output, print loops).
- **Auditability:** job metadata, output, and exit status are recorded.
- **Non-interactive:** no TTY; commands must not prompt for input.

### OS notes (Windows)
- PowerShell does **not** support Bash heredocs (e.g., `python - << EOF`). Use `python -c` or a `.py` file.
- PowerShell may emit benign CLIXML noise to stderr; treat as non-fatal when exit code is `0`.

## Notes

- `.env` is gitignored. Do not commit secrets.
- `FRIDAY_PASSWORD` protects the page/API/WebSocket with a signed cookie.
- The agent has broad shell/filesystem access. Do not expose it to untrusted networks.
- **Conversation memory** (recent turns) persists to `.friday/conversation.json` when `CONVERSATION_MEMORY_ENABLED=true`. It survives server restarts; logout clears it only when `CONVERSATION_MEMORY_CLEAR_ON_LOGOUT=true`. Reset with `DELETE /api/conversation`.
- Long-term **soul memory** is stored in `soul.md` at the agent workdir root (`AGENT_WORKDIR`, or cwd when unset). Friday recalls it every turn and selectively updates it after chats (preferences, behaviors, key learnings — not full transcripts). You can edit `soul.md` directly or reset it with `DELETE /api/soul`. Toggle with `SOUL_ENABLED` / `SOUL_AUTO_UPDATE`.
- Set `AGENT_WORKDIR` to the repo root (e.g. `~/Documents/Iamnotjarvis`) so Friday can reach all sibling projects (`email_agent/`, `web-ui/`, `WOP/`). It defaults to the parent of this `friday/` project folder. `soul.md` and memory files live in this `friday/` folder. Friday can edit its own `friday/friday/` source via file tools when soul or user requests it; **restart** `python -m friday` after Python source changes, or enable `FRIDAY_SELF_RESTART_ENABLED=true` and use the `restart_friday` tool. The former `chat/` project (image/PDF analysis, SerpAPI search) has been merged into Friday as the `analyze_image`, `read_pdf`, and `web_search_serpapi` tools.
- **Token usage** is tracked when `TOKEN_USAGE_ENABLED=true`. Ask Friday in chat ("how many tokens did that use?") or call `GET /api/usage?scope=session|last_turn|lifetime`.
- Optional `QUERY_EXPANSION_ENABLED=true` rewrites vague prompts with the LLM before each agent turn (see `.env.example`). This now runs as part of **task analysis**, which also captures intent, edge cases, and complexity routing.
- **Multi-agent orchestration** (`MULTI_AGENT_ENABLED=true`) deploys sequential explore → execute → verify sub-agents only when task analysis determines the request is complex enough to warrant it. Simple tasks stay on the single-agent path. Set `MULTI_AGENT_ENABLED=false` to disable orchestration while keeping analysis.
- **Autonomous Jarvis mode** (`AUTONOMY_ENABLED=true`, see `.env.example`) runs a background worker with a durable task queue (`.friday/task_queue.json`). User chat is queued by default (`AUTONOMY_QUEUE_USER_TASKS=true`). When a turn hits the step budget, Friday auto-continues up to `AUTONOMY_MAX_CONTINUATIONS` times. When background shell jobs finish, Friday can enqueue follow-up turns (`AUTONOMY_JOB_FOLLOWUP=true`). Status: `GET /api/autonomy`; queue: `GET /api/tasks`.
- **Job watchdog** (`AUTONOMY_WATCHDOG_ENABLED=true`) scans all shell jobs every `AUTONOMY_WATCHDOG_POLL_SECONDS` and stops runaway processes (max runtime, no output, repetitive print loops). It also breaks agent continuation loops and stops all jobs when repeated failing tool calls exceed `AUTONOMY_AGENT_STALL_MAX`. Manual scan: `POST /api/watchdog/inspect`.

## Advanced capabilities (Hermes-style)

**Core**
- **Toolsets** — tools grouped and filtered per role (`FRIDAY_TOOLSETS_ENABLED`); explore/verify sub-agents get restricted sets.
- **Skills** — agentskills.io `skills/*/SKILL.md`; catalog in context, `load_skill` for full instructions.
- **Memory** — `soul.md` (primary), plus `USER.md` and `MEMORY.md`; `GET/DELETE /api/user-memory`, `/api/agent-memory`.
- **Context files** — auto-loads `AGENTS.md`, `.hermes.md`, `CLAUDE.md`, `FRIDAY.md`, `.cursorrules` at turn start.
- **@ references** — `@file`, `@folder/`, `@diff`, `@url` expanded into the message (`REFS_*` env).
- **Checkpoints** — snapshots before file mutations; chat `/rollback` or `POST /api/checkpoints/{id}/rollback`.

**Automation**
- **Cron** — `CRON_ENABLED=true`, jobs in `.friday/cron_jobs.json`, `GET/POST/DELETE /api/cron`.
- **Delegation** — `delegate_task` tool runs up to `DELEGATE_MAX_PARALLEL` sub-agents in parallel.
- **Code exec** — `execute_code` sandbox (`CODE_EXEC_ENABLED=true`).
- **Hooks** — gateway webhooks + plugin guards (`HOOKS_ENABLED`, `.friday/hooks.json`).
- **Batch** — parallel prompts + ShareGPT export (`BATCH_ENABLED`, `/api/batch`).

- On Jetson Nano, use Python 3.8+ (`python3.8 -m friday` if needed).
