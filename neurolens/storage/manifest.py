"""Append-only manifest helpers for the NeuroLens run store."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator

from neurolens.utils.io import ensure_dir

__all__ = ["append_manifest_line", "iter_manifest"]


def append_manifest_line(root: str | Path, row: Dict[str, object]) -> Path:
    """Append ``row`` to the manifest JSONL file under ``root``.

    The function ensures the root directory exists and writes the line atomically by
    appending a serialized JSON object followed by a newline. The resulting file path
    is returned for convenience.
    """

    root_path = ensure_dir(root)
    manifest_path = root_path / "_manifest.jsonl"
    serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
    return manifest_path


def iter_manifest(root: str | Path) -> Iterator[Dict[str, object]]:
    """Yield manifest entries stored under ``root``.

    The manifest is tolerant of trailing newlines or blank records; such entries are
    skipped. Malformed JSON lines raise ``ValueError`` to surface corruption early.
    """

    manifest_path = Path(root) / "_manifest.jsonl"
    if not manifest_path.exists():
        return iter(())

    def _generator() -> Iterator[Dict[str, object]]:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    return _generator()

