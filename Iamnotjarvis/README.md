# Iamnotjarvis

Local web agent powered by Azure OpenAI. It can use host tools for shell commands and filesystem operations with the same permissions as the running Python process.

## Setup

Copy `.env.example` to `.env` and fill:

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
DEPLOYMENT_NAME=
OPENAI_API_VERSION=2024-12-01-preview
PORT=80
HOST=0.0.0.0
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
- The agent has broad shell/filesystem access. Do not expose it to untrusted networks.
- Memory is stored in `agent_memory/` by default and ignored by git.
- On Jetson Nano, use Python 3.8+ (`python3.8 -m friday` if needed).
