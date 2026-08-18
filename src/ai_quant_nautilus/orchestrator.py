"""
Orchestrator: state machine driving the AI quant closed-loop.

Strategy lifecycle:
  generated → backtested → passed → paper → promoted → live → retired

Nautilus integration replaces Freqtrade backtest layer.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ai_quant_nautilus.backtest.nautilus_engine import (
    NautilusBacktestAdapter,
    BacktestOutcome,
)
from ai_quant_nautilus.evaluator.gates import GateResult, EvalResult, GateEvaluator
from ai_quant_nautilus.generator.prompt_builder import (
    GenerationContext,
    build_system_prompt,
    build_user_prompt,
)
from ai_quant_nautilus.generator.schema import STRATEGY_SCHEMA, validate_schema_output
from ai_quant_nautilus.sandbox.ast_guard import ast_guard as sandbox_ast_guard
from ai_quant_nautilus.data.collector import DataCollector

logger = logging.getLogger(__name__)


class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""
    GENERATED = "generated"
    BACKTESTED = "backtested"
    PASSED = "passed"
    PAPER = "paper"
    PROMOTED = "promoted"
    LIVE = "live"
    RETIRED = "retired"


@dataclass
class StrategyRecord:
    """Strategy registry record."""
    id: str
    name: str
    code: str
    nautilus_code: str = ""
    status: StrategyStatus = StrategyStatus.GENERATED
    params: dict = field(default_factory=dict)
    parent_id: Optional[str] = None
    rationale: str = ""
    expected_edge: str = ""

    # Backtest metrics
    backtest_result: dict = field(default_factory=dict)
    backtest_metrics: dict = field(default_factory=dict)
    hyperopt_result: dict = field(default_factory=dict)

    # Evaluation
    eval_passed: bool = False
    eval_reasons: list = field(default_factory=list)

    # Paper trading
    paper_started_at: Optional[float] = None
    paper_metrics: dict = field(default_factory=dict)
    paper_outcome: str = ""

    # Risk allocation
    allocation: dict = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyRecord":
        status_val = data.pop("status", "generated")
        data["status"] = StrategyStatus(status_val)
        return cls(**data)


class Orchestrator:
    """
    Main orchestrator: drives the AI quant closed-loop.

    Replaces Freqtrade with NautilusTrader backtest engine.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        registry_path: Optional[Path] = None,
        llm_client=None,
        nautilus_adapter: Optional[NautilusBacktestAdapter] = None,
    ):
        self.registry_path = registry_path or Path("data/registry.json")
        self.data_dir = data_dir or Path("data/raw")
        self._llm_client = llm_client
        self._nautilus = nautilus_adapter or NautilusBacktestAdapter()
        self._evaluator = GateEvaluator()
        self._strategies: dict[str, StrategyRecord] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
            for s in data.get("strategies", []):
                try:
                    self._strategies[s["id"]] = StrategyRecord.from_dict(s)
                except Exception as e:
                    logger.warning(f"Failed to load strategy {s.get('id')}: {e}")

    def _save_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "strategies": [s.to_dict() for s in self._strategies.values()],
            "updated_at": time.time(),
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def run_iteration(
        self,
        strategy_limit: int = 3,
        market_summary: str = "",
    ) -> dict[str, int]:
        """
        Run one full iteration: generate → backtest → evaluate.

        Returns counts of each stage.
        """
        counts = {"generated": 0, "backtested": 0, "passed": 0}

        # 1. Generate strategies
        new_strategies = self._generate_strategies(strategy_limit, market_summary)
        counts["generated"] = len(new_strategies)
        logger.info(f"Generated {len(new_strategies)} strategies")

        # 2. Backtest with Nautilus
        for strategy in new_strategies:
            outcome = self._backtest(strategy)
            counts["backtested"] += 1

            # 3. Evaluate
            eval_result = self._evaluator.evaluate(outcome)
            strategy.eval_passed = eval_result.passed
            strategy.eval_reasons = [str(g) for g in eval_result.gates]
            strategy.backtest_result = outcome.to_dict()

            if eval_result.passed:
                counts["passed"] += 1
                strategy.status = StrategyStatus.PASSED
            else:
                strategy.status = StrategyStatus.RETIRED

        self._save_registry()
        return counts

    def _generate_strategies(
        self,
        limit: int,
        market_summary: str,
    ) -> list[StrategyRecord]:
        """Generate new strategies via LLM."""
        # Build context from existing strategies
        existing = [s for s in self._strategies.values() if s.status not in (StrategyStatus.RETIRED,)]
        top_strategies = sorted(
            [s for s in existing if s.backtest_metrics],
            key=lambda s: s.backtest_metrics.get("sharpe_ratio", 0),
            reverse=True,
        )[:3]

        ctx = GenerationContext(
            market_summary=market_summary,
            top_strategies=[
                {
                    "name": s.name,
                    "rationale": s.rationale,
                    "sharpe": s.backtest_metrics.get("sharpe_ratio"),
                    "win_rate": s.backtest_metrics.get("win_rate"),
                }
                for s in top_strategies
            ],
            existing_hashes={s.name for s in existing},
        )

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(ctx)

        new_strategies = []
        for i in range(limit):
            if not self._llm_client:
                # Generate mock strategy if no LLM client
                strategy = self._generate_mock_strategy(i)
            else:
                result = self._llm_client.generate(system_prompt, user_prompt)
                valid, errors = validate_schema_output(result)
                if not valid:
                    logger.warning(f"LLM output validation failed: {errors}")
                    continue
                strategy = self._create_strategy_from_llm(result)

            if strategy:
                new_strategies.append(strategy)

        return new_strategies

    def _generate_mock_strategy(self, index: int) -> Optional[StrategyRecord]:
        """Generate a mock strategy for testing without LLM."""
        strategy_name = f"gen_mock_{index:03d}"
        code = f'''
from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce

class {strategy_name}(Strategy):
    """Mock strategy for testing."""

    def __init__(self, config):
        super().__init__(config)
        self._trade_size = Decimal("0.1")

    def on_start(self):
        self.log.info("{strategy_name} started")

    def on_bar(self, bar):
        pass

    def on_stop(self):
        self.log.info("{strategy_name} stopped")
'''
        return StrategyRecord(
            id=str(uuid.uuid4())[:8],
            name=strategy_name,
            code=code,
            rationale="Mock strategy for testing",
            expected_edge="None (test)",
            params={"rsi_window": 14},
        )

    def _create_strategy_from_llm(self, llm_output: dict) -> Optional[StrategyRecord]:
        """Create StrategyRecord from LLM output."""
        name = llm_output.get("name", f"gen_{uuid.uuid4().hex[:6]}")
        code = llm_output.get("code", "")
        params = llm_output.get("params", {})

        # Convert to nautilus format
        nautilus_code, error = translate_strategy(llm_output)
        if error:
            logger.warning(f"Nautilus conversion failed: {error}")
            nautilus_code = code  # fallback

        return StrategyRecord(
            id=str(uuid.uuid4())[:8],
            name=name,
            code=code,
            nautilus_code=nautilus_code,
            rationale=llm_output.get("rationale", ""),
            expected_edge=llm_output.get("expected_edge", ""),
            params=params,
            status=StrategyStatus.GENERATED,
        )

    def _backtest(self, strategy: StrategyRecord) -> BacktestOutcome:
        """Run Nautilus backtest on strategy."""
        # Find available data
        data_files = list(self.data_dir.glob("*.parquet")) + list(self.data_dir.glob("*.csv"))
        data_path = str(data_files[0]) if data_files else None

        outcome = self._nautilus.run_backtest(
            strategy_code=strategy.nautilus_code or strategy.code,
            instrument_id="ETHUSDT.BINANCE",
            data_path=data_path,
        )
        strategy.backtest_metrics = outcome.to_dict()
        return outcome

    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        return self._strategies.get(strategy_id)

    def list_strategies(self, status: Optional[StrategyStatus] = None) -> list[StrategyRecord]:
        if status is None:
            return list(self._strategies.values())
        return [s for s in self._strategies.values() if s.status == status]

    def promote_strategy(self, strategy_id: str) -> bool:
        """Move strategy from PASSED to PROMOTED."""
        strategy = self._strategies.get(strategy_id)
        if not strategy or strategy.status != StrategyStatus.PASSED:
            return False
        strategy.status = StrategyStatus.PROMOTED
        self._save_registry()
        return True

    def start_paper(self, strategy_id: str) -> bool:
        """Start paper trading for a promoted strategy."""
        strategy = self._strategies.get(strategy_id)
        if not strategy or strategy.status != StrategyStatus.PROMOTED:
            return False
        strategy.status = StrategyStatus.PAPER
        strategy.paper_started_at = time.time()
        self._save_registry()
        return True

    def retire_strategy(self, strategy_id: str) -> bool:
        """Retire a strategy."""
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            return False
        strategy.status = StrategyStatus.RETIRED
        self._save_registry()
        return True
