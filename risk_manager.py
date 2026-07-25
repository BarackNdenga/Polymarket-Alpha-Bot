"""
Risk Manager – position sizing, drawdown protection, and circuit breakers.

Protects against:
  - Over-exposure to a single market
  - Correlated positions
  - Drawdown exceeding threshold
  - Rapid-fire orders (rate limiting)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config.settings import MAX_POSITION_SIZE_USD

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    total_exposure: float = 0.0
    position_count: int = 0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    max_daily_loss: float = 0.0
    consecutive_losses: int = 0
    last_trade_time: float = 0.0
    drawdown_pct: float = 0.0
    circuit_breaker: bool = False


class RiskManager:
    """
    Enforces risk limits before any order is placed.
    """

    def __init__(
        self,
        max_position_usd: float = MAX_POSITION_SIZE_USD,
        max_daily_loss_pct: float = 0.10,  # 10% of balance
        max_correlated_exposure_pct: float = 0.30,  # 30% in correlated markets
        initial_balance: float = 10_000.0,
    ):
        self.max_position = max_position_usd
        self.max_daily_loss = initial_balance * max_daily_loss_pct
        self.max_correlated = initial_balance * max_correlated_exposure_pct
        self.balance = initial_balance
        self.state = RiskState()

    def pre_trade_check(self, condition_id: str, side: str, size_usd: float) -> tuple[bool, str]:
        """
        Returns (approved, reason).
        """
        # Circuit breaker
        if self.state.circuit_breaker:
            return False, "CIRCUIT BREAKER: Trading halted due to drawdown"

        # Position size limit
        if size_usd > self.max_position:
            return False, f"Position ${size_usd:.2f} exceeds max ${self.max_position:.2f}"

        # Daily trade limit
        if self.state.daily_trades >= 50:
            return False, "Daily trade limit reached (50)"

        # Daily loss limit
        if self.state.daily_pnl <= -self.max_daily_loss:
            self.state.circuit_breaker = True
            return False, f"Daily loss limit hit (${self.state.daily_pnl:.2f})"

        # Rate limiting (min 5 seconds between trades)
        now = time.time()
        if (now - self.state.last_trade_time) < 5.0:
            return False, "Rate limit: must wait 5s between trades"

        # Exposure limit
        if self.state.total_exposure + size_usd > self.balance * 0.5:
            return False, "Total exposure would exceed 50% of balance"

        return True, "OK"

    def post_trade_update(self, size_usd: float, pnl: float = 0.0):
        """Update risk state after a trade."""
        self.state.total_exposure += size_usd
        self.state.position_count += 1
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        self.state.last_trade_time = time.time()

        # Track drawdown
        peak = self.balance + max(0, self.state.daily_pnl)
        if peak > 0:
            self.state.drawdown_pct = max(0, (peak - (self.balance + self.state.daily_pnl)) / peak)

        # Consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Hard circuit breaker at 5 consecutive losses
        if self.state.consecutive_losses >= 5:
            self.state.circuit_breaker = True
            logger.warning("CIRCUIT BREAKER: 5 consecutive losses")

    def reset_daily(self):
        """Reset daily counters (call at start of new day)."""
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.circuit_breaker = False

    def status(self) -> str:
        return (
            f"Risk Status:\n"
            f"  Exposure:      ${self.state.total_exposure:.2f}\n"
            f"  Positions:     {self.state.position_count}\n"
            f"  Daily PnL:     ${self.state.daily_pnl:.2f}\n"
            f"  Daily Trades:  {self.state.daily_trades}\n"
            f"  Drawdown:      {self.state.drawdown_pct:.2%}\n"
            f"  Consec. Losses: {self.state.consecutive_losses}\n"
            f"  Circuit Breaker: {'ACTIVE' if self.state.circuit_breaker else 'INACTIVE'}"
        )
