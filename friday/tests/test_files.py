from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from friday.runtime.files import workdir_relative


class WorkdirRelativeTests(unittest.TestCase):
    def test_returns_relative_path_inside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            path = workdir / "outputs" / "report.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok", encoding="utf-8")
            self.assertEqual(workdir_relative(workdir, path), "outputs/report.txt")

    def test_returns_absolute_path_outside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "repo"
            workdir.mkdir()
            outside = Path(tmp) / "tmux_boot.sh"
            outside.write_text("#!/bin/bash", encoding="utf-8")
            self.assertEqual(workdir_relative(workdir, outside), str(outside.resolve()))


if __name__ == "__main__":
    unittest.main()
