from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from friday.batch.export import to_sharegpt
from friday.runtime.persist import atomic_write_json


class BatchRunner:
    def __init__(
        self,
        root: Path,
        run_turn: Callable[..., Any],
        max_parallel: int,
        max_items: int,
    ) -> None:
        self.root = root
        self.run_turn = run_turn
        self.max_parallel = max_parallel
        self.max_items = max_items
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def start(self, prompts: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        batch_id = uuid.uuid4().hex[:12]
        capped = prompts[: self.max_items]
        job_dir = self.root / batch_id
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": batch_id,
            "status": "running",
            "total": len(capped),
            "completed": 0,
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        atomic_write_json(job_dir / "manifest.json", manifest)
        self._jobs[batch_id] = manifest
        asyncio.create_task(self._run_batch(batch_id, capped, job_dir))
        return batch_id

    async def _run_batch(self, batch_id: str, prompts: List[str], job_dir: Path) -> None:
        sem = asyncio.Semaphore(self.max_parallel)
        items_dir = job_dir / "items"
        items_dir.mkdir(exist_ok=True)

        async def _one(index: int, prompt: str) -> None:
            async with sem:
                messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
                try:
                    reply, _outputs, _mistakes = await self.run_turn(prompt)
                    messages.append({"role": "assistant", "content": reply})
                except Exception as exc:
                    messages.append({"role": "assistant", "content": f"error: {exc}"})
                item = {"index": index, "prompt": prompt, "sharegpt": to_sharegpt(messages)}
                path = items_dir / f"{index}.json"
                path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                manifest_path = job_dir / "manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["completed"] = manifest.get("completed", 0) + 1
                    atomic_write_json(manifest_path, manifest)

        await asyncio.gather(*[_one(i, p) for i, p in enumerate(prompts)])
        manifest_path = job_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "done"
            atomic_write_json(manifest_path, manifest)
        self._export_jsonl(job_dir, items_dir)

    def _export_jsonl(self, job_dir: Path, items_dir: Path) -> None:
        lines: List[str] = []
        for path in sorted(items_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            lines.append(json.dumps(data.get("sharegpt", {}), ensure_ascii=False))
        (job_dir / "export.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        manifest_path = self.root / batch_id / "manifest.json"
        if not manifest_path.is_file():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def list_batches(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir() and (d / "manifest.json").is_file():
                out.append(json.loads((d / "manifest.json").read_text(encoding="utf-8")))
        return out
