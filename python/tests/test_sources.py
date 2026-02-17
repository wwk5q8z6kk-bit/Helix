"""Tests for Python source connectors."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from core.sources.source_connector import SourceConfig, SourceDocument, SourceRegistry
from core.sources.directory import DirectoryWatcher
from core.sources.rss_feed import RssFeedConnector
from core.sources.github_issues import GitHubIssuesConnector
from core.sources.url_scraper import UrlScraperConnector, extract_text


# --- SourceRegistry ---

class FakeConnector:
    def name(self) -> str:
        return "fake"

    def source_type(self) -> str:
        return "fake"

    async def poll(self):
        return [SourceDocument(title="t", content="c", source_url=None, metadata={}, fetched_at=datetime.now(timezone.utc))]

    async def health_check(self) -> bool:
        return True


def test_registry_register_and_list():
    reg = SourceRegistry()
    cfg = SourceConfig.new("fake", "my-fake")
    reg.register(cfg, FakeConnector())
    configs = reg.list_configs()
    assert len(configs) == 1
    assert configs[0].name == "my-fake"


def test_registry_remove():
    reg = SourceRegistry()
    cfg = SourceConfig.new("fake", "my-fake")
    reg.register(cfg, FakeConnector())
    assert reg.remove(cfg.id) is True
    assert reg.remove(cfg.id) is False
    assert len(reg.list_configs()) == 0


def test_registry_get():
    reg = SourceRegistry()
    cfg = SourceConfig.new("fake", "my-fake")
    conn = FakeConnector()
    reg.register(cfg, conn)
    assert reg.get(cfg.id) is conn
    assert reg.get("nonexistent") is None


async def test_registry_poll():
    reg = SourceRegistry()
    cfg = SourceConfig.new("fake", "my-fake")
    reg.register(cfg, FakeConnector())
    docs = await reg.poll(cfg.id)
    assert len(docs) == 1
    assert docs[0].title == "t"


async def test_registry_poll_missing():
    reg = SourceRegistry()
    with pytest.raises(KeyError):
        await reg.poll("missing")


async def test_registry_health_check():
    reg = SourceRegistry()
    cfg = SourceConfig.new("fake", "my-fake")
    reg.register(cfg, FakeConnector())
    assert await reg.health_check(cfg.id) is True


# --- DirectoryWatcher ---

async def test_directory_watcher_poll(tmp_path):
    # Write a test file
    (tmp_path / "test.txt").write_text("hello world")
    watcher = DirectoryWatcher(str(tmp_path))
    docs = await watcher.poll()
    assert len(docs) == 1
    assert docs[0].title == "test.txt"
    assert "hello world" in docs[0].content


async def test_directory_watcher_health(tmp_path):
    watcher = DirectoryWatcher(str(tmp_path))
    assert await watcher.health_check() is True


async def test_directory_watcher_missing_dir():
    watcher = DirectoryWatcher("/nonexistent/path/abc123")
    assert await watcher.health_check() is False
    docs = await watcher.poll()
    assert docs == []


async def test_directory_watcher_extension_filter(tmp_path):
    (tmp_path / "yes.md").write_text("keep me")
    (tmp_path / "no.txt").write_text("skip me")
    watcher = DirectoryWatcher(str(tmp_path), extensions="md")
    docs = await watcher.poll()
    assert len(docs) == 1
    assert docs[0].title == "yes.md"


# --- RssFeedConnector ---

SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article 1</title>
      <link>https://example.com/1</link>
      <description>Summary one</description>
    </item>
  </channel>
</rss>"""


async def test_rss_feed_poll():
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value=SAMPLE_RSS)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    connector = RssFeedConnector("https://example.com/feed.xml", "test-feed")
    with patch("aiohttp.ClientSession", return_value=mock_session):
        docs = await connector.poll()
    assert len(docs) >= 1
    assert docs[0].title == "Article 1"


async def test_rss_feed_health_check_fail():
    mock_resp = AsyncMock()
    mock_resp.status = 500
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    connector = RssFeedConnector("https://example.com/feed.xml")
    with patch("aiohttp.ClientSession", return_value=mock_session):
        assert await connector.health_check() is False


# --- GitHubIssuesConnector ---

async def test_github_issues_poll():
    issues_json = [
        {
            "title": "Bug report",
            "body": "Something broke",
            "html_url": "https://github.com/org/repo/issues/1",
            "number": 1,
            "state": "open",
            "labels": [{"name": "bug"}],
            "user": {"login": "testuser"},
        }
    ]
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=issues_json)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    connector = GitHubIssuesConnector("org/repo", token="ghp_test123")
    with patch("aiohttp.ClientSession", return_value=mock_session):
        docs = await connector.poll()
    assert len(docs) == 1
    assert docs[0].title == "Bug report"
    assert docs[0].metadata["labels"] == ["bug"]


async def test_github_issues_health_fail():
    mock_resp = AsyncMock()
    mock_resp.status = 404
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    connector = GitHubIssuesConnector("org/nonexistent")
    with patch("aiohttp.ClientSession", return_value=mock_session):
        assert await connector.health_check() is False


# --- UrlScraperConnector ---

async def test_url_scraper_poll():
    html = "<html><body><h1>Title</h1><p>Content here</p></body></html>"
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value=html)
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    connector = UrlScraperConnector("https://example.com/page")
    with patch("aiohttp.ClientSession", return_value=mock_session):
        docs = await connector.poll()
    assert len(docs) == 1
    assert "Title" in docs[0].content
    assert "Content here" in docs[0].content


def test_extract_text():
    html = "<p>Hello</p><script>evil()</script><style>.x{}</style>"
    result = extract_text(html)
    assert "Hello" in result
    assert "evil" not in result
    assert ".x{}" not in result


async def test_url_scraper_empty_urls():
    connector = UrlScraperConnector("")
    docs = await connector.poll()
    assert docs == []


async def test_url_scraper_health():
    connector = UrlScraperConnector("https://example.com")
    assert await connector.health_check() is True
    empty = UrlScraperConnector("")
    assert await empty.health_check() is False
