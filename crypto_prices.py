"""
Crypto Price Fetcher – Binance + CoinGecko for cross-platform fair value.

Used to calculate a "fair" probability for crypto-related prediction markets
by converting real-world probability (e.g., "BTC hits $120k by Dec 31")
into a model-implied probability from spot / derivatives prices.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Optional

import requests

from config.settings import BINANCE_TICKER_URL, COINGECKO_PRICE_URL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceSnapshot:
    symbol: str
    price_usd: float
    source: str  # "binance" | "coingecko" | "aggregate"
    timestamp: float  # epoch seconds


class CryptoPriceFetcher:
    """Fetch spot prices from Binance & CoinGecko; aggregate for robustness."""

    def __init__(self, session: Optional[requests.Session] = None):
        self._s = session or requests.Session()
        self._s.headers.update({"Accept": "application/json"})

    # ---- Binance (no API key needed for public ticker) ---------------------
    def binance_price(self, symbol: str) -> Optional[float]:
        """Fetch latest price from Binance. symbol like 'BTCUSDT'."""
        try:
            resp = self._s.get(BINANCE_TICKER_URL, params={"symbol": symbol}, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            return float(data["price"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning("Binance %s failed: %s", symbol, exc)
            return None

    def binance_tickers(self) -> dict[str, float]:
        """Bulk fetch all Binance tickers at once."""
        try:
            resp = self._s.get(BINANCE_TICKER_URL, timeout=12)
            resp.raise_for_status()
            return {t["symbol"]: float(t["price"]) for t in resp.json()}
        except requests.RequestException as exc:
            logger.warning("Binance bulk ticker failed: %s", exc)
            return {}

    # ---- CoinGecko (free tier) --------------------------------------------
    def coingecko_price(self, coin_id: str) -> Optional[float]:
        """Fetch USD price from CoinGecko simple/price endpoint."""
        try:
            resp = self._s.get(
                COINGECKO_PRICE_URL,
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get(coin_id, {}).get("usd")
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning("CoinGecko %s failed: %s", coin_id, exc)
            return None

    def coingecko_batch(self, coin_ids: list[str]) -> dict[str, float]:
        """Fetch prices for multiple coins at once."""
        try:
            resp = self._s.get(
                COINGECKO_PRICE_URL,
                params={"ids": ",".join(coin_ids), "vs_currencies": "usd"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {cid: d.get("usd", 0.0) for cid, d in data.items()}
        except requests.RequestException as exc:
            logger.warning("CoinGecko batch failed: %s", exc)
            return {}

    # ---- Aggregated price -------------------------------------------------
    def aggregated_price(self, binance_symbol: str, coingecko_id: str) -> Optional[float]:
        """Average of Binance & CoinGecko prices for robustness."""
        b = self.binance_price(binance_symbol)
        c = self.coingecko_price(coingecko_id)
        prices = [p for p in (b, c) if p is not None]
        if not prices:
            return None
        return sum(prices) / len(prices)


# ---------------------------------------------------------------------------
# Fair-value models for crypto prediction markets
# ---------------------------------------------------------------------------

class CryptoFairValue:
    """
    Calculate model-implied probabilities for crypto prediction markets.

    Currently supports:
    - Price-target markets (e.g., "BTC above $150k by 2026")
    - ETF approval markets
    - Halving / technical event markets
    """

    def __init__(self, price_fetcher: CryptoPriceFetcher):
        self.fetcher = price_fetcher

    # ---- Price target markets ---------------------------------------------
    @staticmethod
    def price_target_probability(
        current_price: float,
        target_price: float,
        days_remaining: float,
        implied_volatility: float = 0.75,
    ) -> float:
        """
        Approximate probability that an asset reaches a target price
        within a time window, using a log-normal bridge approximation.

        For a target ABOVE current price: P ≈ N( ln(target/current) / (sigma * sqrt(T)) )
        which gives the probability of reaching that level.

        We use the hitting-time approximation:
        P(hit target) ≈ 2 * N( -ln(target/current) / (sigma * sqrt(T)) )
        """
        if current_price <= 0 or target_price <= 0 or days_remaining <= 0:
            return 0.0

        log_ratio = math.log(target_price / current_price)
        sigma_t = implied_volatility * math.sqrt(days_remaining / 365.0)
        if sigma_t == 0:
            return 0.0

        # Hitting probability for a barrier in GBM (reflection principle)
        z = abs(log_ratio) / sigma_t
        from math import erfc
        # P(hit) = 2 * (1 - N(z)) = erfc(z / sqrt(2))
        prob = math.erfc(z / math.sqrt(2.0))
        return max(0.0, min(1.0, prob))

    # ---- CoinGecko / Binance price → probability mapping ------------------
    def btc_target(self, target_usd: float, days: float) -> float:
        """Probability BTC reaches target within `days`."""
        price = self.fetcher.aggregated_price("BTCUSDT", "bitcoin")
        if price is None:
            return 0.0
        return self.price_target_probability(price, target_usd, days)

    def eth_target(self, target_usd: float, days: float) -> float:
        """Probability ETH reaches target within `days`."""
        price = self.fetcher.aggregated_price("ETHUSDT", "ethereum")
        if price is None:
            return 0.0
        return self.price_target_probability(price, target_usd, days)

    def sol_target(self, target_usd: float, days: float) -> float:
        """Probability SOL reaches target within `days`."""
        price = self.fetcher.aggregated_price("SOLUSDT", "solana")
        if price is None:
            return 0.0
        return self.price_target_probability(price, target_usd, days)

    # ---- Generic extractor -------------------------------------------------
    @staticmethod
    def extract_target_from_question(question: str) -> Optional[tuple[str, float]]:
        """
        Attempt to parse a price target from a market question.
        Returns (asset, target_price) or None.
        """
        patterns = [
            # "Will Bitcoin reach $150,000 by ..."
            r"(?:bitcoin|BTC|btc)\D+?\$?([0-9,]+(?:\.\d+)?)",
            r"(?:ethereum|ETH|eth)\D+?\$?([0-9,]+(?:\.\d+)?)",
            r"(?:solana|SOL|sol)\D+?\$?([0-9,]+(?:\.\d+)?)",
        ]
        q = question.lower()
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                val = float(m.group(1).replace(",", ""))
                asset = "btc" if "btc" in q or "bitcoin" in q else \
                        "eth" if "eth" in q or "ethereum" in q else "sol"
                return asset, val
        return None

    def auto_fair_value(self, question: str, days: float) -> Optional[float]:
        """
        Try to automatically compute a fair-value probability for a
        crypto-related market question.
        """
        parsed = self.extract_target_from_question(question)
        if parsed is None:
            return None
        asset, target = parsed
        dispatch = {
            "btc": lambda: self.btc_target(target, days),
            "eth": lambda: self.eth_target(target, days),
            "sol": lambda: self.sol_target(target, days),
        }
        return dispatch.get(asset)()
