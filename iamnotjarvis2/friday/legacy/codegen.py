from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    return_code: int = 0


@dataclass
class CodeGeneration:
    code: str = ""
    installations: List[str] = field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 30
    error_message: str = ""


@dataclass
class ConversationHistory:
    prompts: List[str] = field(default_factory=list)
    codes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    installations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))


CODE_TEMPLATE = """\
You are a Python code generator. Your task is to create a complete, working Python program
that solves this specific request:
{user_prompt}

Previous Error (if any):
{error_info}

Historical Context:
{context}

Instructions:
1. Use this EXACT structure (this is a template — implement the actual solution):
```python
def main():
    try:
        print("Starting execution...", flush=True)

        # Get user inputs — ALWAYS two separate lines:
        print("Enter ...: ", end="", flush=True)
        variable = input()

        # For numbers, convert after input:
        print("Enter number: ", end="", flush=True)
        number = float(input())

        result = # Your calculation here
        print(f"Result: {{result}}", flush=True)

    except Exception as exc:
        print(f"Error: {{exc}}", flush=True)
        raise
    finally:
        print("Successfully", flush=True)

if __name__ == "__main__":
    main()
```

2. Critical Rules:
   - ALL print statements must include flush=True
   - NEVER combine print and input on the same line
   - ALL numeric inputs must use float() or int() after input()
   - ALWAYS handle errors in try-except
   - ALWAYS include "Successfully" in finally
   - ALWAYS show clear, labelled outputs

3. Format Requirements:
   - 4-space indentation
   - Clear variable names
   - Descriptive prompts

Generate the complete solution for: {user_prompt}
Return ONLY the working Python code with no additional text or explanations.
"""

INSTALL_TEMPLATE = """\
Analyze this Python error and determine required pip packages:

Error: {error_msg}

Previous installations:
{prev_installs}

Rules:
1. Only suggest pip-installable Python packages
2. Only if clearly missing from the error
3. One package per line, format: pip install <package>
4. No system packages (apt, brew, etc.)
5. No already-installed packages

Return only pip install commands, or an empty string if none needed.
"""


class CodeGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.previous_attempts: List[dict] = []

    def generate_code_with_context(
        self,
        prompt: str,
        error_message: str = "",
        historical_context: str = "",
    ) -> str:
        full_prompt = CODE_TEMPLATE.format(
            user_prompt=prompt,
            error_info=error_message or "None",
            context=historical_context or "None",
        )

        if error_message:
            raw = self.llm.chat(
                f"The previous code had this error:\n{error_message}\n\n"
                f"Please fix the code. Remember all the original rules.\n\n"
                f"Original request: {prompt}\n\nReturn ONLY the corrected Python code."
            )
        else:
            self.llm.reset_conversation()
            raw = self.llm.chat(full_prompt)

        cleaned = self._clean_code(raw)
        validated = self._validate_code_structure(cleaned)

        self.previous_attempts.append(
            {
                "prompt": prompt,
                "code": validated,
                "error": error_message,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        return validated

    def generate_installation_commands(
        self, error_message: str, previous_installations: List[str]
    ) -> List[str]:
        if not error_message:
            return []
        prompt = INSTALL_TEMPLATE.format(
            error_msg=error_message,
            prev_installs="\n".join(previous_installations) or "None",
        )
        response = self.llm.invoke(prompt)
        return [
            cmd.strip()
            for cmd in response.splitlines()
            if cmd.strip().startswith("pip install")
            and cmd.strip() not in previous_installations
        ]

    def _clean_code(self, code: str) -> str:
        code = code.strip().replace("\t", "    ")
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        code = code.replace("```python", "").replace("```", "")
        code = code.replace("**name**", "__name__").replace("'__main__'", '"__main__"')

        lines, current_indent = [], 0
        indent = "    "
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if stripped.startswith(("def ", "class ", "if __name__")):
                current_indent = 0
            elif stripped == "try:":
                current_indent = 1
            elif stripped in ["except Exception as exc:", "finally:", "else:"]:
                current_indent = 1
            elif stripped == "main()":
                current_indent = 1
            lines.append(indent * current_indent + stripped)
            if stripped.endswith(":"):
                current_indent += 1

        code = "\n".join(lines)
        return code if code.endswith("\n") else code + "\n"

    def _validate_code_structure(self, code: str) -> str:
        lines, flags = [], {"main": False, "try": False, "except": False, "finally": False}
        indent = "    "
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("def main():"):
                flags["main"] = True
            elif stripped == "try:":
                flags["try"] = True
            elif stripped.startswith("except"):
                flags["except"] = True
            elif stripped == "finally:":
                flags["finally"] = True

            if "input(" in line and "input()" not in line and "=" in line:
                ci = len(line) - len(line.lstrip())
                pad = " " * ci
                var, rest = line.split("=", 1)
                prompt_text = re.search(r"input\((.*?)\)", rest)
                pt = prompt_text.group(1).strip(" '\"") if prompt_text else "Enter value"
                lines.append(f'{pad}print("{pt}", end="", flush=True)')
                if "float(" in line:
                    lines.append(f"{pad}{var.strip()} = float(input())")
                elif "int(" in line:
                    lines.append(f"{pad}{var.strip()} = int(input())")
                else:
                    lines.append(f"{pad}{var.strip()} = input()")
                continue
            lines.append(line)

        if not all(flags.values()):
            return self._fallback_structure()
        return "\n".join(lines)

    def _fallback_structure(self) -> str:
        return (
            'def main():\n'
            '    try:\n'
            '        print("Starting execution...", flush=True)\n'
            '        print("Enter first number: ", end="", flush=True)\n'
            '        num1 = float(input())\n'
            '        print("Enter second number: ", end="", flush=True)\n'
            '        num2 = float(input())\n'
            '        result = num1 + num2\n'
            '        print(f"Result: {result}", flush=True)\n'
            '    except Exception as exc:\n'
            '        print(f"Error: {exc}", flush=True)\n'
            '        raise\n'
            '    finally:\n'
            '        print("Successfully", flush=True)\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n"
        )

    def reset_attempts(self) -> None:
        self.previous_attempts = []


def _detect_terminal() -> Optional[str]:
    for term in [
        "gnome-terminal",
        "xterm",
        "lxterminal",
        "xfce4-terminal",
        "mate-terminal",
        "konsole",
        "terminator",
    ]:
        if shutil.which(term):
            return term
    return None


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class TerminalExecutor:
    def __init__(self, workdir: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.current_dir = workdir
        self.output_dir = os.path.join(self.current_dir, "temp_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.terminal = _detect_terminal() if _has_display() else None
        if self.terminal:
            self.logger.info("GUI terminal detected: %s", self.terminal)
        else:
            self.logger.info("Headless inline execution mode.")

    def execute_in_new_terminal(self, script_path: str, timeout: int = 60) -> ExecutionResult:
        if self.terminal and _has_display():
            return self._execute_gui(script_path, timeout)
        return self._execute_inline(script_path, timeout)

    def _execute_gui(self, script_path: str, timeout: int) -> ExecutionResult:
        base = os.path.splitext(os.path.basename(script_path))[0]
        ts = int(time.time())
        out_f = os.path.join(self.output_dir, f"{base}_{ts}.output")
        err_f = os.path.join(self.output_dir, f"{base}_{ts}.error")
        done_f = os.path.join(self.output_dir, f"{base}_{ts}.complete")
        status_f = os.path.join(self.output_dir, f"{base}_{ts}.status")
        wrapper = os.path.join(self.output_dir, f"{base}_{ts}_wrapper.sh")

        for f in [out_f, err_f, done_f, status_f]:
            if os.path.exists(f):
                os.remove(f)

        script_content = self._build_wrapper(script_path, out_f, err_f, done_f, status_f)
        with open(wrapper, "w", encoding="utf-8") as fh:
            fh.write(script_content)
        os.chmod(wrapper, 0o755)

        try:
            if self.terminal == "gnome-terminal":
                subprocess.Popen([self.terminal, "--", "/bin/bash", wrapper])
            else:
                subprocess.Popen([self.terminal, "-e", f"/bin/bash {wrapper}"])
        except Exception as exc:
            self.logger.error("Failed to open terminal: %s", exc)
            return ExecutionResult(status=ExecutionStatus.ERROR, error_message=str(exc))

        result = self._wait_for_completion(out_f, err_f, done_f, status_f, timeout)
        for f in [out_f, err_f, done_f, status_f, wrapper]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
        return result

    def _execute_inline(self, script_path: str, timeout: int) -> ExecutionResult:
        if script_path.endswith(".py"):
            cmd = [sys.executable, "-u", script_path]
        else:
            cmd = ["/bin/bash", script_path] if sys.platform != "win32" else ["cmd.exe", "/c", script_path]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.current_dir,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
            has_error = (
                rc != 0
                or bool(stderr.strip())
                or re.search(r"(Error|Exception|Traceback)", stdout, re.IGNORECASE)
                or "Successfully" not in stdout
            )
            if has_error:
                err_msg = stderr.strip() or self._extract_errors(stdout) or "Unknown error"
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=err_msg,
                    return_code=rc,
                )
            return ExecutionResult(status=ExecutionStatus.SUCCESS, stdout=stdout, stderr=stderr, return_code=rc)
        except subprocess.TimeoutExpired:
            return ExecutionResult(status=ExecutionStatus.TIMEOUT, error_message="Execution timed out")
        except Exception as exc:
            return ExecutionResult(status=ExecutionStatus.ERROR, error_message=str(exc))

    def run_installation(self, command: str) -> ExecutionResult:
        self.logger.info("Running: %s", command)
        try:
            proc = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0:
                return ExecutionResult(status=ExecutionStatus.SUCCESS, stdout=proc.stdout)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=proc.stderr,
                error_message=proc.stderr or f"Exit code {proc.returncode}",
            )
        except Exception as exc:
            return ExecutionResult(status=ExecutionStatus.ERROR, error_message=str(exc))

    def _build_wrapper(self, script_path: str, out_f: str, err_f: str, done_f: str, status_f: str) -> str:
        py = sys.executable
        return f"""#!/bin/bash
cd "{self.current_dir}"
stdout_tmp="{out_f}.stdout"
stderr_tmp="{out_f}.stderr"

if [[ "{script_path}" == *.sh ]]; then
    bash "{script_path}" 2> >(tee "$stderr_tmp" >&2) | tee "$stdout_tmp"
    exit_code=${{PIPESTATUS[0]}}
else
    "{py}" -u "{script_path}" 2> >(tee "$stderr_tmp" >&2) | tee "$stdout_tmp"
    exit_code=${{PIPESTATUS[0]}}
fi

cat "$stdout_tmp" > "{out_f}"
cat "$stderr_tmp" > "{err_f}" 2>/dev/null || true
rm -f "$stdout_tmp" "$stderr_tmp"

if [ $exit_code -ne 0 ] || grep -qiE "error:|exception|traceback" "{out_f}"; then
    echo "Error occurred (exit $exit_code)" > "{status_f}"
elif grep -q "Successfully" "{out_f}"; then
    echo "Success" > "{status_f}"
else
    echo "Completed without 'Successfully' marker" > "{status_f}"
fi

echo $exit_code > "{done_f}"
echo ""
echo "Press Enter to close..."
read
"""

    def _wait_for_completion(self, out_f: str, err_f: str, done_f: str, status_f: str, timeout: int) -> ExecutionResult:
        start, last_size = time.time(), 0
        while not os.path.exists(done_f):
            if time.time() - start > timeout:
                return ExecutionResult(status=ExecutionStatus.TIMEOUT, error_message="Execution timed out")
            if os.path.exists(out_f):
                try:
                    content = open(out_f, encoding="utf-8").read()
                    if len(content) > last_size:
                        last_size = len(content)
                except OSError:
                    pass
            time.sleep(0.1)

        rc = int(open(done_f, encoding="utf-8").read().strip()) if os.path.exists(done_f) else 1
        stdout = open(out_f, encoding="utf-8").read() if os.path.exists(out_f) else ""
        stderr = open(err_f, encoding="utf-8").read() if os.path.exists(err_f) else ""
        status = open(status_f, encoding="utf-8").read().strip() if os.path.exists(status_f) else ""

        has_error = (
            rc != 0
            or bool(stderr.strip())
            or "Error" in status
            or re.search(r"(Error|Exception|Traceback)", stdout, re.IGNORECASE)
            or "Successfully" not in stdout
        )

        if has_error:
            err_msg = self._extract_errors(stdout) or stderr.strip() or status or "Unknown error"
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stdout=stdout,
                stderr=stderr,
                error_message=err_msg,
                return_code=rc,
            )
        return ExecutionResult(status=ExecutionStatus.SUCCESS, stdout=stdout, stderr=stderr, return_code=rc)

    @staticmethod
    def _extract_errors(text: str) -> str:
        patterns = [
            r"Traceback.*",
            r"(SyntaxError|ImportError|ModuleNotFoundError|TypeError|"
            r"ValueError|NameError|AttributeError|IndexError|KeyError):.*",
        ]
        matches: List[str] = []
        for p in patterns:
            matches += re.findall(p, text, re.IGNORECASE | re.MULTILINE)
        return "\n".join(dict.fromkeys(matches))


class HistoryManager:
    def __init__(self, history_dir: str) -> None:
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save(self, history: ConversationHistory) -> None:
        path = os.path.join(self.history_dir, f"history_{history.timestamp}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": history.timestamp,
                        "prompts": history.prompts,
                        "codes": history.codes,
                        "errors": history.errors,
                        "installations": history.installations,
                    },
                    f,
                    indent=2,
                )
        except OSError as exc:
            self.logger.error("Failed to save history: %s", exc)

    def load_recent(self, limit: int = 5) -> List[ConversationHistory]:
        histories: List[ConversationHistory] = []
        try:
            files = sorted(
                [f for f in os.listdir(self.history_dir) if f.startswith("history_")],
                reverse=True,
            )[:limit]
            for fname in files:
                with open(os.path.join(self.history_dir, fname), encoding="utf-8") as f:
                    d = json.load(f)
                histories.append(
                    ConversationHistory(
                        prompts=d["prompts"],
                        codes=d["codes"],
                        errors=d["errors"],
                        installations=d["installations"],
                        timestamp=d["timestamp"],
                    )
                )
        except OSError as exc:
            self.logger.error("Failed to load history: %s", exc)
        return histories

    def build_context(self, histories: List[ConversationHistory], current_prompt: str) -> str:
        parts, current_words = [], set(current_prompt.lower().split())
        similar: List[Tuple[float, str]] = []
        for h in histories:
            for p, c in zip(h.prompts, h.codes):
                if "Successfully" in c and p:
                    pw = set(p.lower().split())
                    sim = len(current_words & pw) / max(len(current_words | pw), 1)
                    if sim > 0.3:
                        similar.append((sim, f"Similar example:\nPrompt: {p}\nCode:\n{c}"))
        similar.sort(reverse=True)
        parts.extend(s for _, s in similar[:2])
        return "\n\nPrevious successful examples:\n" + "\n\n".join(parts) if parts else ""


class CodeGenerationSystem:
    def __init__(self, llm: LLMClient, workdir: str, emit: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.llm = llm
        self.generator = CodeGenerator(llm)
        self.executor = TerminalExecutor(workdir)
        self.logger = logging.getLogger(__name__)
        self.history_mgr = HistoryManager(os.path.join(workdir, "conversation_history"))
        self.current_history = ConversationHistory()
        self.current_context = ""
        self._emit = emit

    def _log(self, payload: Dict[str, Any]) -> None:
        if self._emit:
            self._emit(payload)

    def run_non_interactive(self, user_prompt: str) -> None:
        generation = CodeGeneration()
        script_path = os.path.join(self.executor.output_dir, "generated_script.py")
        self.current_history.prompts.append(user_prompt)

        if not self.current_context:
            recent = self.history_mgr.load_recent()
            self.current_context = self.history_mgr.build_context(recent, user_prompt)

        while True:
            if generation.attempt_count >= generation.max_attempts:
                self._log({"type": "codegen_log", "text": "Maximum attempts reached."})
                self.history_mgr.save(self.current_history)
                return

            self._log(
                {
                    "type": "codegen_attempt",
                    "attempt": generation.attempt_count + 1,
                }
            )

            session_ctx = "\n\nCurrent session modifications:\n" + "".join(
                f"- {p}\n" for p in self.current_history.prompts[-3:][1:]
            )

            generation.code = self.generator.generate_code_with_context(
                user_prompt,
                generation.error_message,
                self.current_context + session_ctx,
            )
            self.current_history.codes.append(generation.code)
            self._save_script(generation.code, script_path)

            self._log({"type": "codegen_code", "code": generation.code[:12000]})

            result = self.executor.execute_in_new_terminal(script_path)

            if result.error_message:
                self.current_history.errors.append(result.error_message)

            success = (
                result.status == ExecutionStatus.SUCCESS
                and "Successfully" in result.stdout
                and not result.stderr.strip()
                and result.return_code == 0
            )

            if success:
                self._log({"type": "codegen_success", "stdout": result.stdout[:4000]})
                self.history_mgr.save(self.current_history)
                return

            generation.attempt_count += 1
            new_installs = self.generator.generate_installation_commands(
                result.error_message, generation.installations
            )
            for cmd in new_installs:
                ir = self.executor.run_installation(cmd)
                if ir.status == ExecutionStatus.SUCCESS:
                    generation.installations.append(cmd)
                    self.current_history.installations.append(cmd)
                    self._log({"type": "codegen_install", "command": cmd})
                else:
                    self._log({"type": "codegen_install_failed", "error": ir.error_message})

            generation.error_message = result.error_message or result.stderr or "Unknown error"
            self._log({"type": "codegen_error", "message": generation.error_message[:4000]})

    @staticmethod
    def _save_script(code: str, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)


async def run_codegen_non_interactive(
    prompt: str,
    bus: EventBus,
    workdir: str,
    settings: Settings,
) -> None:
    await bus.publish({"type": "codegen_started", "prompt": prompt})
    bag: List[Dict[str, Any]] = []

    def emit(payload: Dict[str, Any]) -> None:
        bag.append(dict(payload))

    def runner() -> None:
        llm = LLMClient(settings=settings)
        system = CodeGenerationSystem(llm, workdir=workdir, emit=emit)
        system.run_non_interactive(prompt)

    await asyncio.to_thread(runner)
    for payload in bag:
        await bus.publish(payload)
    await bus.publish({"type": "codegen_complete"})
