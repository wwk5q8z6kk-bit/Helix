"""RSS/Atom feed source connector."""

from datetime import datetime, timezone
from typing import List, Optional

import aiohttp

from .source_connector import SourceConnector, SourceDocument


class RssFeedConnector:
    """Fetches and parses RSS/Atom feeds."""

    def __init__(self, url: str, feed_name: str = ""):
        self._url = url
        self._feed_name = feed_name or url
        self._last_poll: Optional[datetime] = None

    def name(self) -> str:
        return f"rss:{self._feed_name}"

    def source_type(self) -> str:
        return "rss"

    async def poll(self) -> List[SourceDocument]:
        try:
            import feedparser
        except ImportError:
            return []

        docs: List[SourceDocument] = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self._url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return docs
                    raw = await resp.text()
        except Exception:
            return docs

        feed = feedparser.parse(raw)
        cutoff = self._last_poll
        self._last_poll = datetime.now(timezone.utc)

        for entry in feed.entries:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                from time import mktime
                published = datetime.fromtimestamp(mktime(entry.published_parsed), tz=timezone.utc)

            if cutoff and published and published <= cutoff:
                continue

            content = entry.get("summary", entry.get("description", ""))
            docs.append(SourceDocument(
                title=entry.get("title"),
                content=content,
                source_url=entry.get("link"),
                metadata={"feed": self._feed_name, "author": entry.get("author", "")},
                fetched_at=datetime.now(timezone.utc),
            ))
        return docs

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self._url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status == 200
        except Exception:
            return False
