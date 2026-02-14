"""Interface for persistent memory/storage operations.

Abstracts away database implementation details.
Can be SQLite, PostgreSQL, or other backends.
"""

from typing import Protocol, Optional, Dict, Any, List


class IMemoryStore(Protocol):
    """Interface for persistent memory operations.

    Abstracts away database implementation details.
    Can be SQLite, PostgreSQL, or other backends.
    """

    async def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Store value in memory.

        Args:
            key: Storage key
            value: Value to store (must be JSON-serializable)
            ttl: Optional time-to-live in seconds
        """
        ...

    async def retrieve(
        self,
        key: str
    ) -> Optional[Any]:
        """Retrieve value from memory.

        Args:
            key: Storage key

        Returns:
            Stored value or None if not found
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete a value from memory.

        Args:
            key: Storage key to delete
        """
        ...

    async def list_keys(
        self,
        pattern: Optional[str] = None
    ) -> List[str]:
        """List all keys matching optional pattern.

        Args:
            pattern: Optional glob pattern for keys

        Returns:
            List of matching keys
        """
        ...
