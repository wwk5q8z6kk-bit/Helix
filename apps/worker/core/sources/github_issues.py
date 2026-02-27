"""GitHub Issues source connector."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from .source_connector import SourceConnector, SourceDocument


class GitHubIssuesConnector:
    """Polls GitHub repository issues via the REST API."""

    def __init__(self, repo: str, token: str = ""):
        self._repo = repo  # "owner/repo"
        self._token = token
        self._last_poll: Optional[datetime] = None

    def name(self) -> str:
        return f"github:{self._repo}"

    def source_type(self) -> str:
        return "github_issues"

    async def poll(self) -> List[SourceDocument]:
        docs: List[SourceDocument] = []
        headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        params: Dict[str, str] = {"state": "open", "sort": "updated", "direction": "desc", "per_page": "50"}
        if self._last_poll:
            params["since"] = self._last_poll.isoformat()
        self._last_poll = datetime.now(timezone.utc)

        url = f"https://api.github.com/repos/{self._repo}/issues"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return docs
                    issues = await resp.json()
        except Exception:
            return docs

        for issue in issues:
            if issue.get("pull_request"):
                continue  # skip PRs
            labels = [l["name"] for l in issue.get("labels", [])]
            docs.append(SourceDocument(
                title=issue.get("title", ""),
                content=issue.get("body") or "",
                source_url=issue.get("html_url"),
                metadata={
                    "number": issue.get("number"),
                    "state": issue.get("state"),
                    "labels": labels,
                    "author": issue.get("user", {}).get("login", ""),
                },
                fetched_at=datetime.now(timezone.utc),
            ))
        return docs

    async def health_check(self) -> bool:
        headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        url = f"https://api.github.com/repos/{self._repo}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status == 200
        except Exception:
            return False
