"""
Signal Engine – composite alpha scoring for trade decisions.

Combines four signal sources into a single weighted score:
  1. AI Divergence     – how far human market is from AI consensus
  2. Fair Value Gap    – how far market price is from cross-platform fair value
  3. Whale Momentum    – directional pressure from large orders
  4. Time Decay        – urgency factor based on time to expiry

Only when the composite score exceeds the threshold does the bot place an order.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config.settings import (
    SCORE_THRESHOLD,
    WEIGHTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal types
# ---------------------------------------------------------------------------
class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class ComponentScore:
    """Score from a single signal component."""
    name: str
    value: float  # normalised 0–1
    weight: float
    direction: SignalDirection

    @property
    def weighted(self) -> float:
        return self.value * self.weight


@dataclass
class CompositeSignal:
    """Full scoring output for a market."""
    condition_id: str
    question: str
    side: SignalDirection  # BUY or SELL the YES token
    composite_score: float
    threshold_met: bool
    components: list[ComponentScore]

    # Breakdown
    ai_divergence_score: float
    fair_value_score: float
    whale_momentum_score: float
    time_decay_score: float

    # For logging
    def summary(self) -> str:
        return (
            f"[{self.condition_id}] {self.question}\n"
            f"  Side:     {self.side.value}\n"
            f"  Score:    {self.composite_score:.4f} (threshold {SCORE_THRESHOLD})\n"
            f"  Trigger:  {self.threshold_met}\n"
            f"  AI div:   {self.ai_divergence_score:.4f}\n"
            f"  FV gap:   {self.fair_value_score:.4f}\n"
            f"  Whale:    {self.whale_momentum_score:.4f}\n"
            f"  Time:     {self.time_decay_score:.4f}"
        )


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
class SignalEngine:
    """
    Core scoring engine. Each component produces a 0–1 score that is
    multiplied by its weight and summed to form the composite.
    """

    def __init__(self):
        self.threshold = SCORE_THRESHOLD

    # ---- AI Divergence -----------------------------------------------------
    @staticmethod
    def score_ai_divergence(ai_divergence: float, ai_confidence: float) -> float:
        """
        Higher divergence + higher AI confidence = stronger signal.
        Divergence is clamped at 0.3 (30% gap is extreme).
        """
        divergence_score = min(ai_divergence / 0.30, 1.0)
        # Confidence acts as a multiplier – low confidence dampens the signal
        confidence_factor = 0.3 + 0.7 * min(ai_confidence, 1.0)
        return min(divergence_score * confidence_factor, 1.0)

    # ---- Fair Value Gap ----------------------------------------------------
    @staticmethod
    def score_fair_value_gap(market_prob: float, fair_value: float) -> float:
        """
        Score the gap between the market-implied probability and the
        cross-platform fair value.
        """
        gap = abs(market_prob - fair_value)
        # Gaps > 15% are very strong signals
        gap_score = min(gap / 0.15, 1.0)
        return gap_score

    # ---- Whale Momentum ----------------------------------------------------
    @staticmethod
    def score_whale_momentum(whale_score: float, whale_count: int) -> float:
        """
        Whale score already normalised 0–1. Boost slightly for multiple whales.
        """
        count_boost = min(whale_count / 5.0, 1.0) * 0.2
        return min(whale_score + count_boost, 1.0)

    # ---- Time Decay --------------------------------------------------------
    @staticmethod
    def score_time_decay(time_to_expiry_sec: float) -> float:
        """
        Scoring time urgency:
        - Very short expiry (< 1h) → high urgency (near 1.0)
        - Medium (1h–24h) → moderate (0.3–0.7)
        - Long (> 24h) → low urgency (near 0)

        We want to trade near expiry when mispricings are most exploitable.
        """
        if time_to_expiry_sec <= 0:
            return 0.0

        hours = time_to_expiry_sec / 3600.0

        # Use a smooth exponential decay: urgency is highest near expiry
        import math
        return max(0.05, min(1.0, math.exp(-0.05 * hours) * 1.2))

    # ---- Composite ---------------------------------------------------------
    def compute(
        self,
        condition_id: str,
        question: str,
        market_prob: float,
        ai_divergence: float,
        ai_confidence: float,
        fair_value: Optional[float],
        whale_score: float,
        whale_count: int,
        whale_direction: str,
        time_to_expiry_sec: float,
    ) -> CompositeSignal:
        """
        Compute the full composite signal for a market.
        """
        # 1. AI Divergence
        ai_score = self.score_ai_divergence(ai_divergence, ai_confidence)
        ai_dir = SignalDirection.NONE
        if ai_divergence > 0.05:
            # If AI says probability is HIGHER than market → BUY YES
            ai_dir = SignalDirection.BUY if market_prob < (market_prob + ai_divergence) else SignalDirection.SELL

        # 2. Fair Value Gap
        fv_score = 0.0
        fv_dir = SignalDirection.NONE
        if fair_value is not None:
            fv_score = self.score_fair_value_gap(market_prob, fair_value)
            if fair_value > market_prob:
                fv_dir = SignalDirection.BUY
            elif fair_value < market_prob:
                fv_dir = SignalDirection.SELL

        # 3. Whale Momentum
        w_score = self.score_whale_momentum(whale_score, whale_count)
        w_dir = SignalDirection.NONE
        if whale_score > 0.15:
            w_dir = SignalDirection.BUY if whale_direction == "BUY" else SignalDirection.SELL

        # 4. Time Decay
        t_score = self.score_time_decay(time_to_expiry_sec)

        # Composite
        composite = (
            ai_score * WEIGHTS.ai_divergence
            + fv_score * WEIGHTS.fair_value_gap
            + w_score * WEIGHTS.whale_momentum
            + t_score * WEIGHTS.time_decay
        )

        # Determine final direction via majority vote among non-NONE components
        directions = [d for d in [ai_dir, fv_dir, w_dir] if d != SignalDirection.NONE]
        if directions:
            buy_votes = directions.count(SignalDirection.BUY)
            sell_votes = directions.count(SignalDirection.SELL)
            final_dir = SignalDirection.BUY if buy_votes >= sell_votes else SignalDirection.SELL
        else:
            final_dir = SignalDirection.NONE

        threshold_met = composite >= self.threshold and final_dir != SignalDirection.NONE

        return CompositeSignal(
            condition_id=condition_id,
            question=question,
            side=final_dir,
            composite_score=composite,
            threshold_met=threshold_met,
            components=[
                ComponentScore("AI Divergence", ai_score, WEIGHTS.ai_divergence, ai_dir),
                ComponentScore("Fair Value Gap", fv_score, WEIGHTS.fair_value_gap, fv_dir),
                ComponentScore("Whale Momentum", w_score, WEIGHTS.whale_momentum, w_dir),
                ComponentScore("Time Decay", t_score, WEIGHTS.time_decay, SignalDirection.NONE),
            ],
            ai_divergence_score=ai_score,
            fair_value_score=fv_score,
            whale_momentum_score=w_score,
            time_decay_score=t_score,
        )
