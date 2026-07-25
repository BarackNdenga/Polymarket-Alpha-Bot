"""
Advanced ML Model – Historical Regression for Prediction Markets.

This module builds a machine learning pipeline that learns from historical
Polymarket data to predict whether the current market probability is
mispriced (and in which direction).

Features engineered:
  - Price momentum (3m, 1h, 24h changes)
  - Volume-weighted order flow
  - Spread dynamics (compression/expansion)
  - Time-to-expiry decay curve
  - Whale activity intensity
  - Category-specific baselines (crypto, politics, sports)
  - Cross-platform divergence

Model: Ensemble of Ridge Regression + Gradient Boosting + Logistic Regression.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
@dataclass
class MarketFeatures:
    """Engineered features for a single market evaluation."""
    # Price features
    current_prob: float
    prob_1h_ago: float
    prob_24h_ago: float
    momentum_1h: float
    momentum_24h: float
    volatility_24h: float

    # Order book features
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    bid_depth: float
    ask_depth: float
    book_imbalance: float  # (bid_depth - ask_depth) / total

    # Volume features
    volume_24h: float
    volume_ratio: float  # vs 7-day average
    trade_count_1h: int

    # Time features
    hours_to_expiry: float
    expiry_urgency: float  # 0-1, higher near expiry

    # Whale features
    whale_net_flow: float
    whale_count_1h: int
    whale_momentum: float  # 0-1

    # Cross-platform
    ai_consensus: float
    ai_divergence: float
    ai_confidence: float
    fair_value: Optional[float] = None
    fair_value_gap: float = 0.0

    # Category
    category: str = "general"
    is_crypto: bool = False
    is_politics: bool = False
    is_sports: bool = False

    # Historical accuracy of similar markets
    hist_win_rate: float = 0.5
    hist_avg_score: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Convert features to a numeric vector for ML input."""
        fv = self.fair_value if self.fair_value is not None else self.current_prob
        return np.array([
            self.current_prob,
            self.momentum_1h,
            self.momentum_24h,
            self.volatility_24h,
            self.spread_pct,
            self.book_imbalance,
            self.volume_ratio,
            self.expiry_urgency,
            self.whale_momentum,
            self.ai_divergence,
            self.ai_confidence,
            self.fair_value_gap,
            float(self.is_crypto),
            float(self.is_politics),
            float(self.is_sports),
            self.hist_win_rate,
            self.hist_avg_score,
            self.trade_count_1h,
            self.whale_count_1h,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Simple ML Implementation (no external dependencies)
# ---------------------------------------------------------------------------
class SimpleLinearModel:
    """
    Ridge regression implemented from scratch.
    Used as a lightweight fallback when sklearn is not available.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train via closed-form ridge regression: w = (X^T X + alpha*I)^-1 X^T y"""
        n, d = X.shape
        XtX = X.T @ X + self.alpha * np.eye(d)
        Xty = X.T @ y

        try:
            self.weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to gradient descent
            self.weights = np.zeros(d)
            lr = 0.001
            for _ in range(1000):
                pred = X @ self.weights + self.bias
                grad = X.T @ (pred - y) / n + self.alpha * self.weights
                self.weights -= lr * grad

        self.bias = np.mean(y - X @ self.weights)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(X.shape[0])
        return X @ self.weights + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid-transformed prediction for classification."""
        raw = self.predict(X)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))


class GradientBoostingModel:
    """
    Simple gradient boosting for binary classification.
    No external dependencies.
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 3, lr: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.trees: list = []
        self.base_pred: float = 0.0
        self.is_fitted = False

    def _build_tree(self, residuals: np.ndarray, X: np.ndarray, depth: int = 0) -> dict:
        """Build a simple decision stump/tree."""
        if depth >= self.max_depth or len(residuals) < 10:
            return {"value": float(np.mean(residuals))}

        best_feature = 0
        best_threshold = 0.0
        best_score = float("inf")

        n, d = X.shape
        for feat in range(d):
            values = X[:, feat]
            sorted_idx = np.argsort(values)
            for split in range(1, min(len(sorted_idx), 20)):
                threshold = (values[sorted_idx[split - 1]] + values[sorted_idx[split]]) / 2
                left_mask = values <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue

                left_var = np.var(residuals[left_mask])
                right_var = np.var(residuals[right_mask])
                score = left_mask.sum() * left_var + right_mask.sum() * right_var

                if score < best_score:
                    best_score = score
                    best_feature = feat
                    best_threshold = threshold

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "feature": int(best_feature),
            "threshold": float(best_threshold),
            "left": self._build_tree(residuals[left_mask], X[left_mask], depth + 1),
            "right": self._build_tree(residuals[right_mask], X[right_mask], depth + 1),
        }

    def _predict_tree(self, tree: dict, x: np.ndarray) -> float:
        if "value" in tree:
            return tree["value"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        return self._predict_tree(tree["right"], x)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train gradient boosting."""
        self.base_pred = float(np.mean(y))
        predictions = np.full(len(y), self.base_pred)

        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = self._build_tree(residuals, X)
            self.trees.append(tree)

            # Update predictions
            for i in range(len(y)):
                predictions[i] += self.lr * self._predict_tree(tree, X[i])

        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(X.shape[0], 0.5)
        preds = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            for i in range(X.shape[0]):
                preds[i] += self.lr * self._predict_tree(tree, X[i])
        # Sigmoid
        return 1.0 / (1.0 + np.exp(-np.clip(preds, -10, 10)))


# ---------------------------------------------------------------------------
# ML Pipeline
# ---------------------------------------------------------------------------
class MLSignalModel:
    """
    Full ML pipeline for mispricing detection.

    Architecture:
    1. Feature extraction from market data
    2. Ensemble prediction (Linear + GBM)
    3. Confidence calibration
    4. Persistence (save/load model)
    """

    MODEL_PATH = "data/ml_model.pkl"
    TRAINING_DATA_PATH = "data/training_data.jsonl"

    def __init__(self):
        self.linear_model = SimpleLinearModel(alpha=0.1)
        self.gbm_model = GradientBoostingModel(n_estimators=30, max_depth=3, lr=0.05)
        self.feature_names = [
            "current_prob", "momentum_1h", "momentum_24h", "volatility_24h",
            "spread_pct", "book_imbalance", "volume_ratio", "expiry_urgency",
            "whale_momentum", "ai_divergence", "ai_confidence", "fair_value_gap",
            "is_crypto", "is_politics", "is_sports", "hist_win_rate",
            "hist_avg_score", "trade_count_1h", "whale_count_1h",
        ]
        self._training_samples: list[tuple[np.ndarray, float]] = []

    # ---- Training ----------------------------------------------------------
    def add_sample(self, features: MarketFeatures, outcome: float):
        """
        Add a training sample.
        outcome: 1.0 if the market resolved YES, 0.0 if NO.
        """
        X = features.to_vector()
        self._training_samples.append((X, outcome))

        # Append to file for persistence
        Path(self.TRAINING_DATA_PATH).parent.mkdir(exist_ok=True)
        with open(self.TRAINING_DATA_PATH, "a") as f:
            f.write(json.dumps({
                "features": X.tolist(),
                "outcome": outcome,
                "timestamp": time.time(),
            }) + "\n")

    def train(self):
        """Train the ensemble model on accumulated samples."""
        if len(self._training_samples) < 50:
            logger.warning("Not enough training samples (%d). Need at least 50.", len(self._training_samples))
            return

        X = np.array([s[0] for s in self._training_samples])
        y = np.array([s[1] for s in self._training_samples])

        # Train linear model
        self.linear_model.fit(X, y)

        # Train GBM
        self.gbm_model.fit(X, y)

        # Save
        self.save()
        logger.info("ML Model trained on %d samples. Saved to %s", len(self._training_samples), self.MODEL_PATH)

    def predict_mispricing(self, features: MarketFeatures) -> dict:
        """
        Predict the probability that the market is mispriced
        and the direction of the mispricing.
        """
        X = features.to_vector().reshape(1, -1)

        # Ensemble prediction
        p_linear = self.linear_model.predict_proba(X)[0] if self.linear_model.is_fitted else 0.5
        p_gbm = self.gbm_model.predict_proba(X)[0] if self.gbm_model.is_fitted else 0.5

        # Weighted ensemble (GBM gets more weight if trained well)
        if self.gbm_model.is_fitted:
            ensemble_prob = 0.3 * p_linear + 0.7 * p_gbm
        else:
            ensemble_prob = p_linear

        # Determine direction
        market_vs_model = features.current_prob - ensemble_prob

        if abs(market_vs_model) < 0.03:
            direction = "ALIGNED"
            confidence = 0.0
        elif ensemble_prob > features.current_prob:
            direction = "UNDERPRICED"  # Market too low → BUY
            confidence = min(abs(market_vs_model) / 0.20, 1.0)
        else:
            direction = "OVERPRICED"  # Market too high → SELL
            confidence = min(abs(market_vs_model) / 0.20, 1.0)

        return {
            "ensemble_probability": float(ensemble_prob),
            "linear_probability": float(p_linear),
            "gbm_probability": float(p_gbm),
            "direction": direction,
            "confidence": float(confidence),
            "divergence": float(abs(market_vs_model)),
            "is_mispriced": abs(market_vs_model) > 0.05 and confidence > 0.3,
        }

    # ---- Persistence -------------------------------------------------------
    def save(self):
        """Save trained models to disk."""
        Path(self.MODEL_PATH).parent.mkdir(exist_ok=True)
        data = {
            "linear_weights": self.linear_model.weights.tolist() if self.linear_model.is_fitted else None,
            "linear_bias": self.linear_model.bias,
            "gbm_trees": self.gbm_model.trees if self.gbm_model.is_fitted else [],
            "gbm_base_pred": self.gbm_model.base_pred,
            "feature_names": self.feature_names,
            "n_samples": len(self._training_samples),
        }
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        """Load trained models from disk."""
        if not os.path.exists(self.MODEL_PATH):
            logger.info("No saved model found. Starting fresh.")
            return

        try:
            with open(self.MODEL_PATH, "rb") as f:
                data = pickle.load(f)

            if data.get("linear_weights") is not None:
                self.linear_model.weights = np.array(data["linear_weights"])
                self.linear_model.bias = data["linear_bias"]
                self.linear_model.is_fitted = True

            if data.get("gbm_trees"):
                self.gbm_model.trees = data["gbm_trees"]
                self.gbm_model.base_pred = data["gbm_base_pred"]
                self.gbm_model.is_fitted = True

            logger.info("Loaded ML model (linear=%s, gbm=%s)",
                        "fitted" if self.linear_model.is_fitted else "unfitted",
                        "fitted" if self.gbm_model.is_fitted else "unfitted")
        except Exception as exc:
            logger.error("Failed to load ML model: %s", exc)

    def load_training_data(self):
        """Load training samples from file for retraining."""
        if not os.path.exists(self.TRAINING_DATA_PATH):
            return

        with open(self.TRAINING_DATA_PATH) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        X = np.array(d["features"])
                        y = d["outcome"]
                        self._training_samples.append((X, y))
                    except (json.JSONDecodeError, KeyError):
                        continue

        logger.info("Loaded %d training samples from file", len(self._training_samples))


# ---------------------------------------------------------------------------
# Feature Extractor – converts raw market data into ML features
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """
    Builds MarketFeatures from raw market data, historical prices,
    and signal engine outputs.
    """

    @staticmethod
    def extract(
        market: dict,
        book: dict,
        ai_divergence: float = 0.0,
        ai_confidence: float = 0.5,
        fair_value: Optional[float] = None,
        whale_momentum: float = 0.0,
        whale_count: int = 0,
        whale_net_flow: float = 0.0,
        time_to_expiry_hours: float = 168.0,
        question: str = "",
        # Optional historical context
        prob_1h_ago: Optional[float] = None,
        prob_24h_ago: Optional[float] = None,
        volume_24h: float = 0,
        volume_7d_avg: float = 0,
    ) -> MarketFeatures:
        """Build a complete feature vector from raw inputs."""
        q = question.lower()

        # Price features
        current_prob = (book.get("yes_best_bid", 0) + book.get("yes_best_ask", 1)) / 2
        prob_1h = prob_1h_ago or current_prob
        prob_24h = prob_24h_ago or current_prob
        momentum_1h = current_prob - prob_1h
        momentum_24h = current_prob - prob_24h
        volatility_24h = abs(momentum_1h) + abs(momentum_24h) / 2

        # Order book features
        best_bid = book.get("yes_best_bid", 0.5)
        best_ask = book.get("yes_best_ask", 0.5)
        spread = best_ask - best_bid
        spread_pct = spread / max(current_prob, 0.01)
        bid_depth = sum(float(b.get("size", 0)) for b in book.get("bids", [])[:10])
        ask_depth = sum(float(a.get("size", 0)) for a in book.get("asks", [])[:10])
        total_depth = bid_depth + ask_depth
        book_imbalance = (bid_depth - ask_depth) / max(total_depth, 1)

        # Volume
        vol_ratio = volume_24h / max(volume_7d_avg, 1)

        # Time
        expiry_urgency = max(0.05, min(1.0, math.exp(-0.05 * time_to_expiry_hours) * 1.2))

        # Fair value
        fv = fair_value if fair_value is not None else current_prob
        fv_gap = abs(current_prob - fv)

        # Category detection
        is_crypto = any(kw in q for kw in ["btc", "bitcoin", "eth", "ethereum", "sol", "crypto"])
        is_politics = any(kw in q for kw in ["trump", "biden", "election", "congress", "president"])
        is_sports = any(kw in q for kw in ["nfl", "nba", "mlb", "super bowl", "world cup"])

        if is_crypto:
            category = "crypto"
        elif is_politics:
            category = "politics"
        elif is_sports:
            category = "sports"
        else:
            category = "general"

        return MarketFeatures(
            current_prob=current_prob,
            prob_1h_ago=prob_1h,
            prob_24h_ago=prob_24h,
            momentum_1h=momentum_1h,
            momentum_24h=momentum_24h,
            volatility_24h=volatility_24h,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_pct=spread_pct,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            book_imbalance=book_imbalance,
            volume_24h=volume_24h,
            volume_ratio=vol_ratio,
            trade_count_1h=0,  # Would come from WebSocket
            hours_to_expiry=time_to_expiry_hours,
            expiry_urgency=expiry_urgency,
            whale_net_flow=whale_net_flow,
            whale_count_1h=whale_count,
            whale_momentum=whale_momentum,
            ai_consensus=0.5,  # Would come from AI consensus fetcher
            ai_divergence=ai_divergence,
            ai_confidence=ai_confidence,
            fair_value=fair_value,
            fair_value_gap=fv_gap,
            category=category,
            is_crypto=is_crypto,
            is_politics=is_politics,
            is_sports=is_sports,
        )
