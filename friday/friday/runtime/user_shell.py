"""Build argv / launch mode so host commands run in a profile-aware, user-like shell."""

from __future__ import annotations

import base64
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional

from friday.config import Settings


@dataclass(frozen=True)
class HostLaunch:
    """If exec_argv is set, use create_subprocess_exec; else Windows cmd-style shell."""

    exec_argv: Optional[List[str]]
    use_windows_cmd_shell: bool = False


def _resolve_powershell_exe(kind: str) -> str:
    k = (kind or "powershell").strip().lower()
    if k in ("pwsh", "pwsh.exe"):
        return shutil.which("pwsh") or shutil.which("pwsh.exe") or "pwsh"
    return shutil.which("powershell") or shutil.which("powershell.exe") or "powershell.exe"


def _unix_exec_argv(command: str, settings: Settings) -> List[str]:
    bash = shutil.which("bash")
    if settings.unix_login_shell and bash:
        return [bash, "--login", "-c", command]
    if bash:
        return [bash, "-c", command]
    sh = shutil.which("sh") or "/bin/sh"
    return [sh, "-c", command]


def _windows_powershell_argv(command: str, settings: Settings) -> List[str]:
    exe = _resolve_powershell_exe(settings.windows_shell)
    enc = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return [
        exe,
        "-NoLogo",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        enc,
    ]


def build_host_launch(command: str, settings: Settings) -> HostLaunch:
    """
    Mirror a normal user shell as closely as asyncio allows (no PTY):
    - Windows: PowerShell or pwsh via -EncodedCommand (full env inherited from the parent process).
    - Unix: bash --login -c (optional) so profile PATH/aliases load like an interactive login shell.
    """
    if sys.platform == "win32":
        ws = (settings.windows_shell or "powershell").strip().lower()
        if ws in ("cmd", "comspec", "command"):
            return HostLaunch(exec_argv=None, use_windows_cmd_shell=True)
        return HostLaunch(exec_argv=_windows_powershell_argv(command, settings))
    return HostLaunch(exec_argv=_unix_exec_argv(command, settings))
