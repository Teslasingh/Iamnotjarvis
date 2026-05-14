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
- Memory is session-only. `agent_memory/conversation.jsonl` is cleared on server start, logout, and shutdown.
- On Jetson Nano, use Python 3.8+ (`python3.8 -m friday` if needed).
