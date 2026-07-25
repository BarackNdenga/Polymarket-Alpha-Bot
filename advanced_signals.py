"""
Advanced Signal Enhancers – optional modules that add edge.

1. Multi-Agent Ensemble Scoring
2. Social Sentiment Cross-Reference
3. Cross-Platform Arbitrage Detection (Kalshi, PredictIt, Manifold)
4. Order Book Depth Analysis
5. Historical Cal calibration (Brier Score)
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config.settings import GAMMA_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Multi-Agent Ensemble
# ---------------------------------------------------------------------------
@dataclass
class AgentVote:
    agent_name: str
    probability: float
    confidence: float
    reasoning: str


class MultiAgentEnsemble:
    """
    Simulates a multi-agent ensemble for robust probability estimation.
    In production, each agent would be a different LLM/model.
    """

    def __init__(self):
        self.agents = {
            "bayesian_updater": 0.25,
            "momentum_analyst": 0.20,
            "mean_reversion": 0.20,
            "news_sentiment": 0.15,
            "technical_model": 0.20,
        }

    def ensemble_probability(self, market_prob: float, features: dict) -> tuple[float, float]:
        """
        Returns (ensemble_probability, ensemble_confidence).
        """
        votes: list[tuple[float, float]] = []

        # Bayesian Updater: starts from 0.5, updates with market info
        prior = 0.5
        likelihood = market_prob
        posterior = (prior * likelihood) / (prior * likelihood + (1 - prior) * (1 - likelihood))
        votes.append((posterior, 0.7))

        # Momentum Analyst: follows recent trend
        recent_trend = features.get("recent_trend", 0)
        momentum_prob = market_prob + recent_trend * 0.05
        votes.append((max(0.01, min(0.99, momentum_prob)), 0.5))

        # Mean Reversion: pulls extreme values toward center
        if market_prob > 0.7 or market_prob < 0.3:
            mr_prob = market_prob + (0.5 - market_prob) * 0.15
        else:
            mr_prob = market_prob
        votes.append((mr_prob, 0.6))

        # News Sentiment: uses volume as proxy
        volume_ratio = features.get("volume_ratio", 1.0)
        if volume_ratio > 2.0:
            sentiment_bias = (0.5 - market_prob) * 0.08
        else:
            sentiment_bias = 0
        votes.append((market_prob + sentiment_bias, 0.4))

        # Technical Model: order book shape
        book_imbalance = features.get("book_imbalance", 0)
        tech_prob = market_prob + book_imbalance * 0.03
        votes.append((max(0.01, min(0.99, tech_prob)), 0.55))

        # Weighted average
        weights = list(self.agents.values())
        total_weight = sum(weights)
        ensemble = sum(v * w for v, (_, w) in zip([v[0] for v in votes], zip(votes, weights))) / total_weight
        confidence = sum(c * w for c, (_, w) in zip([v[1] for v in votes], zip(votes, weights))) / total_weight

        return ensemble, confidence


# ---------------------------------------------------------------------------
# 2. Order Book Depth Analysis
# ---------------------------------------------------------------------------
class OrderBookAnalyzer:
    """
    Analyze order book depth for liquidity signals.
    """

    @staticmethod
    def liquidity_score(bids: list[dict], asks: list[dict]) -> dict:
        """
        Returns liquidity metrics for the order book.
        """
        bid_depth = sum(float(b.get("size", 0)) for b in bids[:10])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:10])

        total = bid_depth + ask_depth
        if total == 0:
            return {"score": 0, "ratio": 0.5, "bid_depth": 0, "ask_depth": 0}

        ratio = bid_depth / total  # >0.5 = buy pressure
        imbalance = abs(ratio - 0.5) * 2  # 0 = balanced, 1 = extreme

        # Higher total depth = better liquidity = lower signal (less opportunity)
        depth_score = max(0, 1.0 - total / 1000.0)

        return {
            "liquidity_score": depth_score,
            "imbalance": imbalance,
            "bid_ratio": ratio,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
        }


# ---------------------------------------------------------------------------
# 3. Cross-Platform Arbitrage
# ---------------------------------------------------------------------------
class ArbitrageDetector:
    """
    Detect price discrepancies across prediction market platforms.
    """

    # Platform mappings
    PLATFORMS = {
        "polymarket": {"base_url": GAMMA_URL, "scale": 1.0},
        "kalshi": {"base_url": "https://api.elections.kalshi.com/trade-api/v2", "scale": 1.0},
        "manifold": {"base_url": "https://api.manifold.markets/v0", "scale": 1.0},
    }

    def __init__(self, session: Optional[requests.Session] = None):
        self._s = session or requests.Session()

    def detect_arbitrage(self, condition_id: str, poly_probability: float) -> Optional[dict]:
        """
        Compare Polymarket probability with other platforms.
        Returns arbitrage opportunity if gap > 5%.
        """
        opportunities = []

        # Kalshi comparison (simplified)
        try:
            resp = self._s.get(
                f"{self.PLATFORMS['kalshi']['base_url']}/markets/{condition_id}",
                timeout=8,
            )
            if resp.status_code == 200:
                kalshi_data = resp.json()
                kalshi_prob = float(kalshi_data.get("result_probability", 0))
                gap = abs(poly_probability - kalshi_prob)
                if gap > 0.05:
                    opportunities.append({
                        "platform": "kalshi",
                        "probability": kalshi_prob,
                        "gap": gap,
                        "direction": "BUY_POLY" if poly_probability < kalshi_prob else "SELL_POLY",
                    })
        except (requests.RequestException, KeyError, ValueError):
            pass

        # Manifold comparison
        try:
            resp = self._s.get(
                f"{self.PLATFORMS['manifold']['base_url']}/market/{condition_id}",
                timeout=8,
            )
            if resp.status_code == 200:
                manifold_data = resp.json()
                manifold_prob = float(manifold_data.get("probability", 0))
                gap = abs(poly_probability - manifold_prob)
                if gap > 0.05:
                    opportunities.append({
                        "platform": "manifold",
                        "probability": manifold_prob,
                        "gap": gap,
                        "direction": "BUY_POLY" if poly_probability < manifold_prob else "SELL_POLY",
                    })
        except (requests.RequestException, KeyError, ValueError):
            pass

        if opportunities:
            return {
                "condition_id": condition_id,
                "poly_probability": poly_probability,
                "opportunities": opportunities,
                "best_gap": max(o["gap"] for o in opportunities),
            }
        return None


# ---------------------------------------------------------------------------
# 4. Historical Calibration (Brier Score Tracking)
# ---------------------------------------------------------------------------
class CalibrationTracker:
    """
    Track prediction accuracy using Brier score.
    Helps tune the model over time.
    """

    def __init__(self, storage_path: str = "logs/calibration.jsonl"):
        self.storage_path = storage_path
        self.records: list[dict] = []
        self._load()

    def _load(self):
        import json
        try:
            with open(self.storage_path) as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
        except FileNotFoundError:
            pass

    def record_prediction(self, condition_id: str, predicted_prob: float):
        self.records.append({
            "condition_id": condition_id,
            "predicted_prob": predicted_prob,
            "timestamp": time.time(),
            "resolved": False,
        })
        self._save()

    def mark_resolved(self, condition_id: str, outcome: bool):
        for r in self.records:
            if r["condition_id"] == condition_id and not r["resolved"]:
                r["actual"] = 1.0 if outcome else 0.0
                r["brier"] = (predicted_prob := r["predicted_prob"]) - r["actual"]
                r["brier"] = r["brier"] ** 2
                r["resolved"] = True
        self._save()

    def _save(self):
        import json
        with open(self.storage_path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")

    def current_brier_score(self) -> Optional[float]:
        resolved = [r for r in self.records if r.get("resolved")]
        if not resolved:
            return None
        return sum(r["brier"] for r in resolved) / len(resolved)

    def summary(self) -> str:
        brier = self.current_brier_score()
        total = len(self.records)
        resolved = sum(1 for r in self.records if r.get("resolved"))
        return f"Predictions: {total} | Resolved: {resolved} | Brier: {brier or 'N/A'}"
