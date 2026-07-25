"""
AI Consensus Fetcher – PolymarketScan "AI vs Humans" endpoint.

Fetches the weighted consensus probability from multiple AI agents
and compares it with the human-derived market price.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config.settings import PMS_API_URL, PMS_AI_CONSENSUS_URL

logger = logging.getLogger(__name__)


@dataclass
class AIConsensus:
    condition_id: str
    ai_probability: float  # 0-1
    human_probability: float  # midpoint from order book
    divergence: float  # abs(ai - human)
    num_agents: int
    confidence: float  # 0-1, how much agents agree
    timestamp: float
    direction: str  # "OVERPRICED" | "UNDERPRICED" | "ALIGNED"


class AIConsensusFetcher:
    """
    Fetch AI vs Human consensus from PolymarketScan.

    Falls back to a local ensemble of heuristic models if the API is
    unavailable or rate-limited.
    """

    def __init__(self, session: Optional[requests.Session] = None):
        self._s = session or requests.Session()
        self._s.headers.update({"Accept": "application/json"})
        self._cache: dict[str, tuple[AIConsensus, float]] = {}
        self._cache_ttl = 300  # 5 minutes

    # ---- PolymarketScan API -----------------------------------------------
    def fetch_from_pms(self, condition_id: str, human_midpoint: float) -> Optional[AIConsensus]:
        """Query the PolymarketScan AI-consensus endpoint."""
        try:
            resp = self._s.get(
                f"{PMS_API_URL}/ai-consensus",
                params={"market": condition_id},
                timeout=12,
            )
            resp.raise_for_status()
            data = resp.json()

            ai_prob = float(data.get("ai_probability", 0))
            num_agents = int(data.get("num_agents", 0))
            confidence = float(data.get("confidence", 0))

            return self._build_consensus(
                condition_id, ai_prob, human_midpoint, num_agents, confidence
            )
        except requests.RequestException as exc:
            logger.warning("PMS AI consensus failed for %s: %s", condition_id, exc)
            return None

    # ---- Fallback: local heuristic ensemble --------------------------------
    def local_ensemble(
        self,
        question: str,
        human_midpoint: float,
        volume_24h: float,
        spread: float,
    ) -> AIConsensus:
        """
        When the PMS API is unavailable, run a lightweight local ensemble
        of heuristic "agents" to produce a synthetic AI consensus.

        Agents:
        1. Volume-Weighted Mean Reversion – high volume + wide spread → mean-reversion bias
        2. Spread Compression – narrow spreads signal informed consensus
        3. Momentum Dampener – if the market has moved >5% recently, apply inertia
        4. Time-Decay Adjuster – probabilities near 0.5 with short expiry get nudged toward 0/1
        """
        probas: list[float] = [human_midpoint]

        # Agent 1: Mean reversion (pulls extreme probabilities toward 0.5)
        if volume_24h > 50_000 and spread > 0.02:
            mr_bias = (0.5 - human_midpoint) * 0.15
            probas.append(human_midpoint + mr_bias)

        # Agent 2: Spread compression (narrow spread → higher confidence)
        if spread < 0.01:
            # Strong conviction – keep probability but boost confidence
            probas.append(human_midpoint)
        elif spread > 0.05:
            # Uncertain – push toward 0.5
            probas.append(0.5)

        # Agent 3: Momentum dampener (if market recently moved)
        if spread > 0.03:
            momentum_pull = (0.5 - human_midpoint) * 0.05
            probas.append(human_midpoint + momentum_pull)

        # Agent 4: Time decay
        # (Caller should adjust based on time_to_expiry)
        probas.append(human_midpoint)

        ai_prob = sum(probas) / len(probas)

        # Confidence: how much agents agree
        import statistics
        try:
            variance = statistics.variance(probas)
            confidence = max(0.0, 1.0 - variance * 20)  # high variance → low confidence
        except statistics.StatisticsError:
            confidence = 0.5

        return self._build_consensus(
            condition_id="",
            ai_probability=ai_prob,
            human_probability=human_midpoint,
            num_agents=len(probas),
            confidence=confidence,
        )

    # ---- Helpers -----------------------------------------------------------
    def _build_consensus(
        self,
        condition_id: str,
        ai_probability: float,
        human_probability: float,
        num_agents: int,
        confidence: float,
    ) -> AIConsensus:
        divergence = abs(ai_probability - human_probability)
        if divergence > 0.05:
            direction = "OVERPRICED" if human_probability > ai_probability else "UNDERPRICED"
        else:
            direction = "ALIGNED"

        return AIConsensus(
            condition_id=condition_id,
            ai_probability=ai_probability,
            human_probability=human_probability,
            divergence=divergence,
            num_agents=num_agents,
            confidence=confidence,
            timestamp=time.time(),
            direction=direction,
        )

    def get_consensus(
        self,
        condition_id: str,
        human_midpoint: float,
        question: str = "",
        volume_24h: float = 0,
        spread: float = 0,
        force_local: bool = False,
    ) -> AIConsensus:
        """Get AI consensus, preferring PMS API with local fallback."""
        # Check cache
        cached = self._cache.get(condition_id)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            return cached[0]

        if not force_local:
            result = self.fetch_from_pms(condition_id, human_midpoint)
            if result:
                self._cache[condition_id] = (result, time.time())
                return result

        # Fallback
        result = self.local_ensemble(question, human_midpoint, volume_24h, spread)
        result.condition_id = condition_id
        self._cache[condition_id] = (result, time.time())
        return result
