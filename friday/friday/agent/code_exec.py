from __future__ import annotations

import json
import secrets
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from friday.config import Settings
from friday.paths import SANDBOX_DIR

_RPC_BOOTSTRAP = textwrap.dedent(
    '''
    import json, sys, urllib.request
    TOKEN = {token!r}
    BASE = {base!r}
    def _call(tool, args):
        body = json.dumps({{"tool": tool, "args": args, "token": TOKEN}}).encode()
        req = urllib.request.Request(BASE, data=body, method="POST",
            headers={{"Content-Type": "application/json"}})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    class friday_tools:
        @staticmethod
        def run(name, **kwargs):
            return _call(name, kwargs)
    '''
)


class CodeExecSession:
    def __init__(self, settings: Settings, workdir: Path) -> None:
        self.settings = settings
        self.workdir = workdir
        self._tokens: Dict[str, int] = {}
        self._call_counts: Dict[str, int] = {}

    def create_token(self, run_id: str) -> str:
        token = secrets.token_hex(16)
        self._tokens[token] = self.settings.code_exec_max_tool_calls
        self._call_counts[run_id] = 0
        return token

    def validate_and_consume(self, token: str) -> bool:
        remaining = self._tokens.get(token)
        if remaining is None or remaining <= 0:
            return False
        self._tokens[token] = remaining - 1
        return True

    def allowed_tools(self) -> set:
        raw = self.settings.code_exec_allowed_tools or ""
        return {t.strip() for t in raw.split(",") if t.strip()}


async def run_execute_code(
    code: str,
    *,
    settings: Settings,
    workdir: Path,
) -> str:
    if not settings.code_exec_enabled:
        return json.dumps({"error": "execute_code disabled; set CODE_EXEC_ENABLED=true"})
    run_id = secrets.token_hex(8)
    sandbox = SANDBOX_DIR / run_id
    sandbox.mkdir(parents=True, exist_ok=True)
    script_path = sandbox / "script.py"
    script_path.write_text(code, encoding="utf-8")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(sandbox),
            capture_output=True,
            text=True,
            timeout=settings.code_exec_max_seconds,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "execute_code timeout", "run_id": run_id})
    elapsed = time.time() - start
    return json.dumps(
        {
            "run_id": run_id,
            "return_code": proc.returncode,
            "stdout": (proc.stdout or "")[-50000:],
            "stderr": (proc.stderr or "")[-20000:],
            "elapsed_seconds": round(elapsed, 2),
        },
        ensure_ascii=False,
    )
