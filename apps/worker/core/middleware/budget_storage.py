"""Persistent storage for LLM budget data."""

import json
import logging
import os
from datetime import date
from typing import Dict

logger = logging.getLogger(__name__)

class PersistentBudgetStore:
    def __init__(self, storage_path: str = "~/.helix/budget.json"):
        self.storage_path = os.path.expanduser(storage_path)
        self._ensure_directory()

    def _ensure_directory(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def load(self) -> Dict[str, float]:
        """Load daily spend from disk."""
        if not os.path.exists(self.storage_path):
            return {}

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            # Check if data is for today
            today = date.today().isoformat()
            if data.get("date") == today:
                return data.get("spend", {})
            else:
                logger.info("Outdated budget data on disk for %s, starting fresh for %s", data.get("date"), today)
                return {}
        except Exception as e:
            logger.error("Failed to load budget data: %s", e)
            return {}

    def save(self, spend: Dict[str, float]):
        """Save daily spend to disk."""
        try:
            data = {
                "date": date.today().isoformat(),
                "spend": spend
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save budget data: %s", e)
