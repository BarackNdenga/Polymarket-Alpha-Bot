"""
Whale Shadowing – detect and score large-order momentum.

Tracks whale activity from Polymarket's trade stream and the Data API,
computing a directional momentum score.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import requests

from config.settings import DATA_API_URL, WHALE_THRESHOLD_USD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class WhaleTrade:
    market_condition_id: str
    side: str  # "BUY" | "SELL"
    price: float
    size: float  # number of shares
    usd_value: float
    wallet: str
    timestamp: float
    token_id: str

    @property
    def is_whale(self) -> bool:
        return self.usd_value >= WHALE_THRESHOLD_USD


@dataclass
class WhaleSignal:
    """Aggregated whale momentum for a single market."""
    condition_id: str
    net_flow_usd: float  # positive = buy pressure
    buy_volume_usd: float
    sell_volume_usd: float
    whale_count: int
    dominant_side: str  # "BUY" | "SELL" | "NEUTRAL"
    score: float  # 0.0 – 1.0 normalised momentum
    age_sec: float  # how old the oldest whale trade is


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class WhaleTracker:
    """
    Monitor large trades and compute a directional momentum score per market.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        whale_threshold: float = WHALE_THRESHOLD_USD,
        lookback_sec: float = 1800.0,  # 30 minutes
    ):
        self._s = session or requests.Session()
        self._threshold = whale_threshold
        self._lookback = lookback_sec
        # In-memory trade cache: condition_id -> deque of WhaleTrade
        self._trades: dict[str, deque[WhaleTrade]] = {}

    # ---- Fetch recent trades -----------------------------------------------
    def refresh(self, condition_id: str) -> list[WhaleTrade]:
        """Pull latest trades from the Data API and update local cache."""
        url = f"{DATA_API_URL}/trades"
        params = {
            "market": condition_id,
            "_limit": 200,
            "_sort": "timestamp",
            "_order": "desc",
        }
        try:
            resp = self._s.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
        except requests.RequestException:
            raw = []

        trades: deque[WhaleTrade] = self._trades.get(condition_id, deque(maxlen=500))
        seen = {t.token_id + str(t.timestamp) for t in trades}

        for t in raw:
            tid = str(t.get("token_id", ""))
            ts = float(t.get("timestamp", 0))
            key = tid + str(ts)
            if key in seen:
                continue
            seen.add(key)

            price = float(t.get("price", 0))
            size = float(t.get("size", 0))
            usd = price * size
            side = t.get("maker_side", "BUY")  # Data API convention
            trades.append(WhaleTrade(
                market_condition_id=condition_id,
                side=side,
                price=price,
                size=size,
                usd_value=usd,
                wallet=t.get("maker_address", ""),
                timestamp=ts,
                token_id=tid,
            ))

        self._trades[condition_id] = trades
        return [t for t in trades if t.usd_value >= self._threshold]

    # ---- Signal computation ------------------------------------------------
    def compute_signal(self, condition_id: str) -> WhaleSignal:
        """Compute a whale momentum score for the market."""
        self.refresh(condition_id)
        trades = self._trades.get(condition_id, deque())

        now = time.time()
        recent = [t for t in trades if (now - t.timestamp) <= self._lookback]
        whales = [t for t in recent if t.is_whale]

        buy_usd = sum(t.usd_value for t in whales if t.side == "BUY")
        sell_usd = sum(t.usd_value for t in whales if t.side == "SELL")
        net = buy_usd - sell_usd
        total = buy_usd + sell_usd

        if total == 0:
            score = 0.0
            dominant = "NEUTRAL"
        else:
            # Normalise: stronger net flow → higher score, dampened by log
            import math
            score = min(1.0, math.tanh(abs(net) / (total + 1e-6) * 2.0))
            dominant = "BUY" if net > 0 else "SELL"

        oldest = min((t.timestamp for t in recent), default=now)
        age = now - oldest

        return WhaleSignal(
            condition_id=condition_id,
            net_flow_usd=net,
            buy_volume_usd=buy_usd,
            sell_volume_usd=sell_usd,
            whale_count=len(whales),
            dominant_side=dominant,
            score=score,
            age_sec=age,
        )

    # ---- Quick lookup ------------------------------------------------------
    def is_whale_active(self, condition_id: str) -> bool:
        sig = self.compute_signal(condition_id)
        return sig.whale_count > 0 and sig.score > 0.15
