"""
Polymarket Data Fetcher – retrieves live market state from the CLOB & Gamma APIs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

from config.settings import CLOB_URL, GAMMA_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass
class MarketBook:
    """Snapshot of a single binary outcome market."""
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    end_date_iso: Optional[str] = None
    active: bool = True

    # Order-book snapshot
    yes_best_bid: Optional[float] = None
    yes_best_ask: Optional[float] = None
    no_best_bid: Optional[float] = None
    no_best_ask: Optional[float] = None
    yes_best_bid_size: float = 0.0
    yes_best_ask_size: float = 0.0
    no_best_bid_size: float = 0.0
    no_best_ask_size: float = 0.0

    # Derived
    @property
    def midpoint(self) -> float:
        if self.yes_best_bid and self.yes_best_ask:
            return (self.yes_best_bid + self.yes_best_ask) / 2
        return 0.0

    @property
    def spread(self) -> float:
        if self.yes_best_bid and self.yes_best_ask:
            return self.yes_best_ask - self.yes_best_bid
        return float("inf")

    @property
    def volume_24h(self) -> float:
        return 0.0  # filled by caller if available

    def time_to_expiry(self) -> float:
        """Return seconds until market end_date (negative if expired)."""
        if not self.end_date_iso:
            return float("inf")
        try:
            end = datetime.fromisoformat(self.end_date_iso.replace("Z", "+00:00"))
            return (end - datetime.now(end.tzinfo)).total_seconds()
        except Exception:
            return float("inf")

    @property
    def min_order_size(self) -> int:
        return 5  # Polymarket default


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------
class PolymarketDataFetcher:
    """Thin wrapper around Gamma + CLOB REST APIs."""

    def __init__(self, session: Optional[requests.Session] = None):
        self._s = session or requests.Session()
        self._s.headers.update({"Accept": "application/json"})

    # ---- Gamma (market discovery) ------------------------------------------
    def discover_markets(self, limit: int = 50) -> list[dict]:
        """Fetch the most actively traded markets right now."""
        url = f"{GAMMA_URL}/markets"
        params = {"active": "true", "_sort": "volume24hr", "_order": "desc", "_limit": limit}
        try:
            resp = self._s.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Gamma discover_markets failed: %s", exc)
            return []

    def market_details(self, condition_id: str) -> Optional[dict]:
        """Fetch detailed metadata for a single market."""
        url = f"{GAMMA_URL}/markets/{condition_id}"
        try:
            resp = self._s.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("market_details(%s) failed: %s", condition_id, exc)
            return None

    # ---- CLOB (order book) -------------------------------------------------
    def order_book(self, token_id: str) -> dict:
        """Return the raw CLOB order book for a given outcome token."""
        url = f"{CLOB_URL}/book"
        params = {"token_id": token_id}
        try:
            resp = self._s.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data
        except requests.RequestException as exc:
            logger.warning("order_book(%s) failed: %s", token_id, exc)
            return {}

    def best_prices(self, token_id: str) -> tuple[Optional[float], Optional[float]]:
        """Return (best_bid, best_ask) from the CLOB."""
        book = self.order_book(token_id)
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None
        best_bid_size = float(bids[0]["size"]) if bids else 0.0
        best_ask_size = float(asks[0]["size"]) if asks else 0.0
        return best_bid, best_ask, best_bid_size, best_ask_size

    def midpoint(self, token_id: str) -> float:
        bid, ask, _, _ = self.best_prices(token_id)
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return 0.0

    # ---- Market snapshot builder -------------------------------------------
    def build_market_book(self, market: dict) -> Optional[MarketBook]:
        """Combine Gamma metadata with live CLOB data into a MarketBook."""
        tokens = market.get("clobTokenIds")
        if not tokens:
            return None
        import json
        try:
            token_ids = json.loads(tokens) if isinstance(tokens, str) else tokens
        except (json.JSONDecodeError, TypeError):
            return None

        yes_tid, no_tid = token_ids[0], token_ids[1]

        yes_bid, yes_ask, yes_bsz, yes_asz = self.best_prices(yes_tid)
        no_bid, no_ask, no_bsz, no_asz = self.best_prices(no_tid)

        return MarketBook(
            condition_id=market.get("conditionId", ""),
            question=market.get("question", ""),
            yes_token_id=yes_tid,
            no_token_id=no_tid,
            end_date_iso=market.get("endDate"),
            active=market.get("active", True),
            yes_best_bid=yes_bid,
            yes_best_ask=yes_ask,
            no_best_bid=no_bid,
            no_best_ask=no_ask,
            yes_best_bid_size=yes_bsz,
            yes_best_ask_size=yes_asz,
            no_best_bid_size=no_bsz,
            no_best_ask_size=no_asz,
        )

    # ---- Trades / whales ---------------------------------------------------
    def recent_trades(self, condition_id: str, limit: int = 100) -> list[dict]:
        """Fetch the latest trades for a market from the Data API."""
        from config.settings import DATA_API_URL
        url = f"{DATA_API_URL}/trades"
        params = {"market": condition_id, "_limit": limit, "_sort": "timestamp", "_order": "desc"}
        try:
            resp = self._s.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return []
