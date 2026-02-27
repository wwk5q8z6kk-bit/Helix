"""URL scraper source connector."""

import re
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp

from .source_connector import SourceConnector, SourceDocument


def extract_text(html: str) -> str:
    """Strip HTML tags using regex (no BeautifulSoup dependency)."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class UrlScraperConnector:
    """Scrapes content from a list of URLs."""

    def __init__(self, urls: str, selector: str = ""):
        self._urls = [u.strip() for u in urls.split(",") if u.strip()]
        self._selector = selector  # reserved for future CSS selector support

    def name(self) -> str:
        count = len(self._urls)
        return f"scraper:{count}-urls"

    def source_type(self) -> str:
        return "url_scraper"

    async def poll(self) -> List[SourceDocument]:
        docs: List[SourceDocument] = []
        if not self._urls:
            return docs

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                for url in self._urls:
                    try:
                        async with session.get(url) as resp:
                            if resp.status != 200:
                                continue
                            html = await resp.text()
                            content = extract_text(html)
                            docs.append(SourceDocument(
                                title=url.split("/")[-1] or url,
                                content=content,
                                source_url=url,
                                metadata={"status": resp.status, "content_type": resp.headers.get("content-type", "")},
                                fetched_at=datetime.now(timezone.utc),
                            ))
                    except Exception:
                        continue
        except Exception:
            pass
        return docs

    async def health_check(self) -> bool:
        return len(self._urls) > 0
