"""Budget tracking for LLM API costs with per-provider daily caps."""

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BudgetAction(Enum):
    ALLOW = "allow"
    REJECT = "reject"
    DOWNGRADE = "downgrade"


@dataclass
class ProviderCostConfig:
    name: str
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    daily_budget: float = 100.0


@dataclass
class UsageRecord:
    provider: str
    input_tokens: int
    output_tokens: int
    cost: float
    task_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class BudgetTracker:
    def __init__(self, daily_budget: float = 100.0, default_action: str = "downgrade"):
        self._lock = threading.Lock()
        self.daily_budget = daily_budget
        self.default_action = BudgetAction(default_action)
        self._today: date = datetime.now(tz=timezone.utc).date()
        self._daily_spend: Dict[str, float] = {}
        self._usage_history: List[UsageRecord] = []
        self._provider_configs: Dict[str, ProviderCostConfig] = {}
        self._initialize_default_costs()

    def _initialize_default_costs(self):
        """Initialize default cost configs for known providers."""
        defaults = {
            "claude-opus-4-6": (0.015, 0.075, 30.0),
            "claude-sonnet-4-5": (0.003, 0.015, 50.0),
            "claude-haiku-4-5": (0.0008, 0.004, 80.0),
            "gpt-5.3-codex": (0.010, 0.030, 40.0),
            "gpt-4.1-mini": (0.0004, 0.0016, 80.0),
            "gemini-3-flash": (0.0003, 0.001, 90.0),
            "gemini-3-pro": (0.007, 0.021, 40.0),
            "grok-4": (0.005, 0.015, 40.0),
            "deepseek-r1": (0.0014, 0.0056, 60.0),
            "openrouter": (0.004, 0.012, 50.0),
            "ollama": (0.0, 0.0, 999.0),
            "mistral-large-latest": (0.004, 0.012, 50.0),
            "together": (0.0009, 0.0009, 70.0),
            "fireworks": (0.0009, 0.0009, 70.0),
            "perplexity": (0.005, 0.015, 40.0),
            "cohere": (0.003, 0.015, 50.0),
            "bedrock": (0.008, 0.024, 40.0),
            "cloudflare": (0.0005, 0.002, 80.0),
            "venice": (0.001, 0.004, 70.0),
            "custom": (0.002, 0.008, 50.0),
        }
        for name, (inp, out, budget) in defaults.items():
            self._provider_configs[name] = ProviderCostConfig(
                name=name,
                cost_per_1k_input_tokens=inp,
                cost_per_1k_output_tokens=out,
                daily_budget=budget,
            )

    def _check_date_reset(self):
        """Reset daily spend counters if the date has changed."""
        today = datetime.now(tz=timezone.utc).date()
        if today != self._today:
            self._daily_spend.clear()
            self._today = today
            logger.info("Budget tracker daily reset for %s", today.isoformat())

    def estimate_cost(self, provider: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost for a given provider and token count."""
        cfg = self._provider_configs.get(provider)
        if cfg is None:
            return 0.0
        return (input_tokens / 1000) * cfg.cost_per_1k_input_tokens + (output_tokens / 1000) * cfg.cost_per_1k_output_tokens

    def check_budget(self, provider: str, estimated_cost: float) -> BudgetAction:
        """Check whether a request should be allowed, rejected, or downgraded."""
        with self._lock:
            self._check_date_reset()

            total_spend = sum(self._daily_spend.values())
            if total_spend + estimated_cost > self.daily_budget:
                return self.default_action

            cfg = self._provider_configs.get(provider)
            if cfg is not None:
                provider_spend = self._daily_spend.get(provider, 0.0)
                if provider_spend + estimated_cost > cfg.daily_budget:
                    return BudgetAction.DOWNGRADE

            return BudgetAction.ALLOW

    def record_usage(
        self, provider: str, input_tokens: int, output_tokens: int, task_type: str = "general"
    ) -> UsageRecord:
        """Record a completed LLM call and its cost."""
        cost = self.estimate_cost(provider, input_tokens, output_tokens)
        record = UsageRecord(
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            task_type=task_type,
        )
        with self._lock:
            self._check_date_reset()
            self._daily_spend[provider] = self._daily_spend.get(provider, 0.0) + cost
            self._usage_history.append(record)
        logger.debug("Recorded usage: provider=%s cost=%.6f", provider, cost)
        return record

    def suggest_cheaper_provider(self, required_quality: float = 7.0) -> Optional[str]:
        """Suggest a cheaper provider that meets the quality threshold."""
        from core.llm.intelligent_llm_router import IntelligentLLMRouter

        stats = IntelligentLLMRouter.DEFAULT_PROVIDER_STATS
        if not stats:
            # First router hasn't been created yet — trigger class-level init
            IntelligentLLMRouter()
            stats = IntelligentLLMRouter.DEFAULT_PROVIDER_STATS

        candidates = []
        for provider, pstats in stats.items():
            if pstats.quality_score >= required_quality:
                cfg = self._provider_configs.get(provider.value)
                cost = cfg.cost_per_1k_input_tokens if cfg else pstats.cost_per_1k_tokens
                candidates.append((provider.value, cost, pstats.quality_score))

        if not candidates:
            return None

        # Sort by cost ascending, then quality descending
        candidates.sort(key=lambda c: (c[1], -c[2]))
        return candidates[0][0]

    def get_dashboard(self) -> Dict[str, Any]:
        """Return a summary of current budget state."""
        with self._lock:
            self._check_date_reset()
            total_spend = sum(self._daily_spend.values())
            return {
                "daily_budget": self.daily_budget,
                "total_spent_today": round(total_spend, 6),
                "budget_remaining": round(self.daily_budget - total_spend, 6),
                "date": self._today.isoformat(),
                "provider_spend": {k: round(v, 6) for k, v in self._daily_spend.items()},
                "total_requests_today": sum(
                    1 for r in self._usage_history if r.timestamp.date() == self._today
                ),
                "provider_configs": {
                    name: {
                        "cost_per_1k_input": cfg.cost_per_1k_input_tokens,
                        "cost_per_1k_output": cfg.cost_per_1k_output_tokens,
                        "daily_budget": cfg.daily_budget,
                    }
                    for name, cfg in self._provider_configs.items()
                },
            }
