"""Directory watcher source connector."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .source_connector import SourceConnector, SourceDocument


class DirectoryWatcher:
    """Watches a directory for files modified since last poll."""

    def __init__(self, path: str, extensions: str = ""):
        self._path = Path(path)
        self._extensions = [e.strip() for e in extensions.split(",") if e.strip()] if extensions else []
        self._last_poll: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def name(self) -> str:
        return f"directory:{self._path}"

    def source_type(self) -> str:
        return "directory"

    async def poll(self) -> List[SourceDocument]:
        docs: List[SourceDocument] = []
        if not self._path.is_dir():
            return docs

        cutoff = self._last_poll
        self._last_poll = datetime.now(timezone.utc)

        for entry in self._path.iterdir():
            if not entry.is_file():
                continue
            if self._extensions and entry.suffix.lstrip(".") not in self._extensions:
                continue
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
            if mtime > cutoff:
                try:
                    content = entry.read_text(errors="replace")
                except Exception:
                    content = ""
                docs.append(SourceDocument(
                    title=entry.name,
                    content=content,
                    source_url=str(entry.absolute()),
                    metadata={"size": entry.stat().st_size, "extension": entry.suffix},
                    fetched_at=datetime.now(timezone.utc),
                ))
        return docs

    async def health_check(self) -> bool:
        return self._path.is_dir()
