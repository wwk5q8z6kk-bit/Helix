from typing import Protocol, runtime_checkable, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class SourceDocument:
    title: Optional[str]
    content: str
    source_url: Optional[str]
    metadata: Dict[str, Any]
    fetched_at: datetime


@runtime_checkable
class SourceConnector(Protocol):
    def name(self) -> str: ...
    def source_type(self) -> str: ...
    async def poll(self) -> List[SourceDocument]: ...
    async def health_check(self) -> bool: ...


@dataclass
class SourceConfig:
    id: str
    source_type: str
    name: str
    enabled: bool = True
    settings: Dict[str, str] = field(default_factory=dict)
    poll_interval_secs: int = 300
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def new(source_type: str, name: str) -> "SourceConfig":
        return SourceConfig(
            id=str(uuid.uuid4()),
            source_type=source_type,
            name=name,
        )


class SourceRegistry:
    """In-memory registry mirroring the Rust SourceRegistry pattern."""

    def __init__(self) -> None:
        self._sources: Dict[str, tuple[SourceConfig, SourceConnector]] = {}

    def register(self, config: SourceConfig, connector: SourceConnector) -> None:
        self._sources[config.id] = (config, connector)

    def remove(self, source_id: str) -> bool:
        return self._sources.pop(source_id, None) is not None

    def get(self, source_id: str) -> Optional[SourceConnector]:
        entry = self._sources.get(source_id)
        return entry[1] if entry else None

    def list_configs(self) -> List[SourceConfig]:
        return [config for config, _ in self._sources.values()]

    async def poll(self, source_id: str) -> List[SourceDocument]:
        entry = self._sources.get(source_id)
        if entry is None:
            raise KeyError(f"source {source_id} not found")
        _, connector = entry
        return await connector.poll()

    async def health_check(self, source_id: str) -> bool:
        entry = self._sources.get(source_id)
        if entry is None:
            raise KeyError(f"source {source_id} not found")
        _, connector = entry
        return await connector.health_check()
