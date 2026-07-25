"""
Smart Limit Order Executor – intelligent order placement with dynamic spread.

Never places dumb market orders. Instead:
  - Calculates optimal bid/ask based on order book depth
  - Adjusts spread dynamically based on time-to-expiry and volatility
  - Implements paper-trading mode for risk-free validation
  - Tracks PnL in real-time
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from config.settings import MAX_POSITION_SIZE_USD, MAX_SPREAD_PCT, PAPER_BALANCE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order types
# ---------------------------------------------------------------------------
class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    LIVE = "LIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class ExecutionMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass
class OrderParams:
    """Parameters for a limit order."""
    token_id: str
    side: OrderSide
    price: float
    size: float  # number of shares
    expiration: Optional[int] = None  # Unix timestamp for GTD


@dataclass
class OrderResult:
    """Result of an order placement."""
    order_id: str
    status: OrderStatus
    price: float
    size: float
    cost_usd: float
    timestamp: float
    mode: ExecutionMode


# ---------------------------------------------------------------------------
# Spread calculator
# ---------------------------------------------------------------------------
class DynamicSpreadCalculator:
    """
    Calculate optimal bid/ask spread based on:
    - Current order book depth
    - Time to expiry (closer expiry → tighter spread)
    - Signal strength (stronger signal → more aggressive)
    - Recent volatility proxy (spread width)
    """

    def __init__(self, max_spread_pct: float = MAX_SPREAD_PCT):
        self.max_spread_pct = max_spread_pct

    def optimal_price(
        self,
        side: OrderSide,
        best_bid: Optional[float],
        best_ask: Optional[float],
        signal_score: float,
        time_to_expiry_sec: float,
        current_spread: float,
    ) -> float:
        """
        Calculate the optimal limit price.

        Strategy:
        - BUY:  place bid between best_bid and midpoint, closer to midpoint for strong signals
        - SELL: place ask between best_ask and midpoint, closer to midpoint for strong signals

        The aggressiveness scales with signal strength and time pressure.
        """
        if best_bid is None or best_ask is None:
            # No book – fall back to a midpoint estimate
            return 0.5

        midpoint = (best_bid + best_ask) / 2.0
        book_spread = best_ask - best_bid

        # Don't place orders with spread wider than max
        if book_spread > self.max_spread_pct:
            logger.warning("Spread too wide (%.4f), skipping order", book_spread)
            return -1.0  # sentinel: do not place

        # Aggressiveness factor: 0.0 (passive) → 1.0 (aggressive)
        signal_factor = min(signal_score, 1.0)

        # Time factor: closer to expiry → more aggressive
        hours = time_to_expiry_sec / 3600.0
        if hours < 1:
            time_factor = 0.8
        elif hours < 6:
            time_factor = 0.5
        elif hours < 24:
            time_factor = 0.3
        else:
            time_factor = 0.1

        aggressiveness = (signal_factor * 0.6 + time_factor * 0.4)

        if side == OrderSide.BUY:
            # Place bid between best_bid and midpoint
            bid = best_bid + aggressiveness * book_spread * 0.7
            # Ensure we don't cross the spread
            bid = min(bid, midpoint - book_spread * 0.05)
            return round(bid, 4)
        else:
            # Place ask between midpoint and best_ask
            ask = best_ask - aggressiveness * book_spread * 0.7
            ask = max(ask, midpoint + book_spread * 0.05)
            return round(ask, 4)

    def calculate_size(self, usd_amount: float, price: float, min_order_size: int = 5) -> float:
        """Calculate share quantity from USD amount, respecting min order size."""
        if price <= 0:
            return min_order_size
        shares = math.floor(usd_amount / price)
        return max(float(min_order_size), shares)


# ---------------------------------------------------------------------------
# Paper trading engine
# ---------------------------------------------------------------------------
@dataclass
class PaperPosition:
    """A paper trading position."""
    token_id: str
    side: OrderSide
    entry_price: float
    shares: float
    current_price: float
    pnl: float
    entry_time: float

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100


class PaperTradingEngine:
    """
    Simulates order execution and tracks PnL without real funds.
    """

    def __init__(self, initial_balance: float = PAPER_BALANCE):
        self.balance = initial_balance
        self.positions: list[PaperPosition] = []
        self.order_history: list[OrderResult] = []
        self.total_pnl = 0.0
        self.trades_executed = 0

    def execute(self, params: OrderParams) -> OrderResult:
        """Simulate order execution."""
        cost = params.price * params.size
        if cost > self.balance:
            logger.warning("Paper: insufficient balance ($%.2f) for order ($%.2f)", self.balance, cost)
            return OrderResult(
                order_id=f"PAPER-{int(time.time())}",
                status=OrderStatus.CANCELLED,
                price=params.price,
                size=params.size,
                cost_usd=cost,
                timestamp=time.time(),
                mode=ExecutionMode.PAPER,
            )

        self.balance -= cost
        self.positions.append(PaperPosition(
            token_id=params.token_id,
            side=params.side,
            entry_price=params.price,
            shares=params.size,
            current_price=params.price,
            pnl=0.0,
            entry_time=time.time(),
        ))
        self.trades_executed += 1

        result = OrderResult(
            order_id=f"PAPER-{int(time.time())}",
            status=OrderStatus.FILLED,
            price=params.price,
            size=params.size,
            cost_usd=cost,
            timestamp=time.time(),
            mode=ExecutionMode.PAPER,
        )
        self.order_history.append(result)
        logger.info(
            "PAPER EXECUTION: %s %s shares @ %.4f | Cost: $%.2f | Remaining: $%.2f",
            params.side.value, params.size, params.price, cost, self.balance,
        )
        return result

    def update_prices(self, prices: dict[str, float]):
        """Update current prices and recalculate PnL for all positions."""
        for pos in self.positions:
            if pos.token_id in prices:
                pos.current_price = prices[pos.token_id]
                pos.pnl = (pos.current_price - pos.entry_price) * pos.shares

        self.total_pnl = sum(p.pnl for p in self.positions)

    def summary(self) -> str:
        return (
            f"Paper Trading Summary:\n"
            f"  Balance:     ${self.balance:.2f}\n"
            f"  Positions:   {len(self.positions)}\n"
            f"  Total PnL:   ${self.total_pnl:.2f}\n"
            f"  Trades:      {self.trades_executed}"
        )


# ---------------------------------------------------------------------------
# Live order executor
# ---------------------------------------------------------------------------
class LiveOrderExecutor:
    """
    Execute orders on the real Polymarket CLOB.
    Requires valid authentication (L1 private key or L2 API key pair).
    """

    def __init__(self, client=None):
        """
        client: an authenticated Polymarket CLOB client instance
                (py-clob-client or custom authenticated session).
        """
        self.client = client
        self.order_history: list[OrderResult] = []

    def execute(self, params: OrderParams) -> OrderResult:
        """Place a limit order on the real CLOB."""
        if self.client is None:
            logger.error("Live executor not initialised with an authenticated client")
            return OrderResult(
                order_id="ERROR",
                status=OrderStatus.CANCELLED,
                price=params.price,
                size=params.size,
                cost_usd=params.price * params.size,
                timestamp=time.time(),
                mode=ExecutionMode.LIVE,
            )

        try:
            # Using py-clob-client-v2 API
            response = self.client.place_limit_order(
                token_id=params.token_id,
                side=params.side.value,
                price=str(params.price),
                size=str(params.size),
                expiration=params.expiration,
            )

            result = OrderResult(
                order_id=str(response.order_id) if hasattr(response, "order_id") else "UNKNOWN",
                status=OrderStatus.LIVE if hasattr(response, "ok") and response.ok else OrderStatus.CANCELLED,
                price=params.price,
                size=params.size,
                cost_usd=params.price * params.size,
                timestamp=time.time(),
                mode=ExecutionMode.LIVE,
            )
            self.order_history.append(result)
            logger.info(
                "LIVE ORDER: %s %s shares @ %.4f | ID: %s | Status: %s",
                params.side.value, params.size, params.price, result.order_id, result.status.value,
            )
            return result

        except Exception as exc:
            logger.error("Live order failed: %s", exc)
            return OrderResult(
                order_id="ERROR",
                status=OrderStatus.CANCELLED,
                price=params.price,
                size=params.size,
                cost_usd=params.price * params.size,
                timestamp=time.time(),
                mode=ExecutionMode.LIVE,
            )


# ---------------------------------------------------------------------------
# Unified executor (routes to paper or live)
# ---------------------------------------------------------------------------
class OrderExecutor:
    """Routes orders to paper or live executor based on config."""

    def __init__(
        self,
        mode: ExecutionMode,
        paper_engine: Optional[PaperTradingEngine] = None,
        live_executor: Optional[LiveOrderExecutor] = None,
    ):
        self.mode = mode
        self.paper = paper_engine or PaperTradingEngine()
        self.live = live_executor or LiveOrderExecutor()
        self.spread_calc = DynamicSpreadCalculator()
        self.trade_log: list[dict] = []
        self.last_trade_time: dict[str, float] = {}  # cooldown tracker

    def can_trade(self, condition_id: str, cooldown_sec: int = 300) -> bool:
        """Check if enough cooldown has passed since last trade on this market."""
        from config.settings import TRADE_COOLDOWN_SEC
        last = self.last_trade_time.get(condition_id, 0)
        return (time.time() - last) >= TRADE_COOLDOWN_SEC

    def place_order(
        self,
        token_id: str,
        side: OrderSide,
        best_bid: Optional[float],
        best_ask: Optional[float],
        signal_score: float,
        time_to_expiry_sec: float,
        current_spread: float,
        condition_id: str,
        max_usd: float = MAX_POSITION_SIZE_USD,
    ) -> Optional[OrderResult]:
        """Full pipeline: calculate optimal price/size → place order."""
        if not self.can_trade(condition_id):
            logger.info("Cooldown active for %s, skipping", condition_id[:12])
            return None

        # Calculate optimal price
        price = self.spread_calc.optimal_price(
            side=side,
            best_bid=best_bid,
            best_ask=best_ask,
            signal_score=signal_score,
            time_to_expiry_sec=time_to_expiry_sec,
            current_spread=current_spread,
        )
        if price < 0:
            logger.info("Spread too wide, skipping order")
            return None

        # Validate price is within [0.01, 0.99]
        price = max(0.01, min(0.99, price))

        # Calculate size
        size = self.spread_calc.calculate_size(max_usd, price)

        # Set GTD expiration: 30 minutes from now (minimum 3 min)
        expiration = int(time.time()) + 60 + 1800  # 30 min

        params = OrderParams(
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            expiration=expiration,
        )

        # Execute
        if self.mode == ExecutionMode.PAPER:
            result = self.paper.execute(params)
        else:
            result = self.live.execute(params)

        if result.status == OrderStatus.FILLED or result.status == OrderStatus.LIVE:
            self.last_trade_time[condition_id] = time.time()
            self.trade_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "condition_id": condition_id,
                "side": side.value,
                "price": price,
                "size": size,
                "score": signal_score,
                "mode": self.mode.value,
                "order_id": result.order_id,
            })

        return result

    def performance_report(self) -> str:
        """Generate a performance summary."""
        if self.mode == ExecutionMode.PAPER:
            return self.paper.summary()
        return f"Live trades: {len(self.trade_log)}\n" + "\n".join(
            f"  {t['timestamp']} | {t['side']} @ {t['price']:.4f} | {t['size']} shares"
            for t in self.trade_log[-10:]
        )
