"""
Unit tests for the ML Model.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signals.ml_model import (
    SimpleLinearModel,
    GradientBoostingModel,
    MLSignalModel,
    FeatureExtractor,
    MarketFeatures,
)


def test_linear_model():
    """Test ridge regression from scratch."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    w = np.array([1.0, -0.5, 0.3, 0.8, -0.2])
    y = X @ w + 0.5 + np.random.randn(100) * 0.1

    model = SimpleLinearModel(alpha=0.01)
    model.fit(X, y)
    assert model.is_fitted

    preds = model.predict(X[:5])
    assert len(preds) == 5
    # Predictions should be close to actual
    actual = X[:5] @ w + 0.5
    assert np.allclose(preds, actual, atol=0.5), f"Preds: {preds}, Actual: {actual}"
    print(f"  Linear model test PASSED")


def test_gbm_model():
    """Test gradient boosting from scratch."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(float)  # Simple binary target

    model = GradientBoostingModel(n_estimators=20, max_depth=2, lr=0.1)
    model.fit(X, y)
    assert model.is_fitted

    probs = model.predict_proba(X[:10])
    assert all(0 <= p <= 1 for p in probs), "Probabilities must be in [0,1]"
    print(f"  GBM model test PASSED")


def test_ensemble_prediction():
    """Test the full ML pipeline."""
    ml = MLSignalModel()

    # Generate synthetic training data
    np.random.seed(42)
    for _ in range(60):
        feat = MarketFeatures(
            current_prob=np.random.uniform(0.1, 0.9),
            prob_1h_ago=np.random.uniform(0.1, 0.9),
            prob_24h_ago=np.random.uniform(0.1, 0.9),
            momentum_1h=np.random.uniform(-0.1, 0.1),
            momentum_24h=np.random.uniform(-0.2, 0.2),
            volatility_24h=np.random.uniform(0, 0.3),
            best_bid=0.5,
            best_ask=0.5,
            spread=0.02,
            spread_pct=0.04,
            bid_depth=100,
            ask_depth=100,
            book_imbalance=np.random.uniform(-0.3, 0.3),
            volume_24h=10000,
            volume_ratio=1.0,
            trade_count_1h=20,
            hours_to_expiry=168,
            expiry_urgency=0.3,
            whale_net_flow=0,
            whale_count_1h=0,
            whale_momentum=0,
            ai_consensus=0.5,
            ai_divergence=np.random.uniform(0, 0.3),
            ai_confidence=np.random.uniform(0.3, 0.9),
            fair_value=None,
            fair_value_gap=0,
            category="general",
            is_crypto=False,
            is_politics=False,
            is_sports=False,
        )
        outcome = 1.0 if feat.current_prob > 0.5 else 0.0
        ml.add_sample(feat, outcome)

    ml.train()
    assert ml.gbm_model.is_fitted, "GBM should be trained"
    assert ml.linear_model.is_fitted, "Linear should be trained"

    # Predict on a new sample
    test_feat = MarketFeatures(
        current_prob=0.3,
        prob_1h_ago=0.35,
        prob_24h_ago=0.4,
        momentum_1h=-0.05,
        momentum_24h=-0.1,
        volatility_24h=0.15,
        best_bid=0.28,
        best_ask=0.32,
        spread=0.04,
        spread_pct=0.13,
        bid_depth=50,
        ask_depth=150,
        book_imbalance=-0.5,
        volume_24h=50000,
        volume_ratio=2.5,
        trade_count_1h=40,
        hours_to_expiry=48,
        expiry_urgency=0.6,
        whale_net_flow=3000,
        whale_count_1h=3,
        whale_momentum=0.7,
        ai_consensus=0.5,
        ai_divergence=0.2,
        ai_confidence=0.8,
        fair_value=0.52,
        fair_value_gap=0.22,
        category="crypto",
        is_crypto=True,
        is_politics=False,
        is_sports=False,
    )
    pred = ml.predict_mispricing(test_feat)
    assert "ensemble_probability" in pred
    assert "direction" in pred
    assert "confidence" in pred
    assert "is_mispriced" in pred
    print(f"  Ensemble prediction test PASSED: {pred}")


def test_feature_extractor():
    """Test feature extraction from raw data."""
    feat = FeatureExtractor.extract(
        market={"question": "Will BTC reach $200k by Dec 2026?"},
        book={
            "yes_best_bid": 0.34,
            "yes_best_ask": 0.37,
            "bids": [{"price": "0.34", "size": "100"}, {"price": "0.33", "size": "50"}],
            "asks": [{"price": "0.37", "size": "80"}, {"price": "0.38", "size": "60"}],
        },
        ai_divergence=0.22,
        ai_confidence=0.78,
        fair_value=0.52,
        whale_momentum=0.65,
        whale_count=4,
        whale_net_flow=4200,
        time_to_expiry_hours=168,
        question="Will BTC reach $200k by Dec 31, 2026?",
    )
    vec = feat.to_vector()
    assert len(vec) == 19, f"Expected 19 features, got {len(vec)}"
    assert feat.is_crypto, "Should detect crypto category"
    assert feat.category == "crypto"
    assert feat.ai_divergence == 0.22
    assert feat.fair_value_gap == abs(0.355 - 0.52)
    print(f"  Feature extractor test PASSED ({len(vec)} features)")


def test_persistence():
    """Test model save/load."""
    import tempfile
    ml = MLSignalModel()
    ml.MODEL_PATH = "/tmp/test_model.pkl"

    # Train
    np.random.seed(42)
    for _ in range(55):
        feat = MarketFeatures(
            current_prob=np.random.uniform(0.1, 0.9),
            prob_1h_ago=0.5, prob_24h_ago=0.5,
            momentum_1h=0, momentum_24h=0,
            volatility_24h=0.1,
            best_bid=0.5, best_ask=0.5,
            spread=0.02, spread_pct=0.04,
            bid_depth=100, ask_depth=100,
            book_imbalance=0,
            volume_24h=10000, volume_ratio=1,
            trade_count_1h=20,
            hours_to_expiry=168, expiry_urgency=0.3,
            whale_net_flow=0, whale_count_1h=0, whale_momentum=0,
            ai_consensus=0.5, ai_divergence=0.1, ai_confidence=0.6,
            fair_value=None, fair_value_gap=0,
        )
        ml.add_sample(feat, 1.0 if feat.current_prob > 0.5 else 0.0)
    ml.train()

    # Save and reload
    ml2 = MLSignalModel()
    ml2.load()
    assert ml2.gbm_model.is_fitted, "Reloaded model should be fitted"
    print(f"  Persistence test PASSED")

    # Cleanup
    os.remove(ml.MODEL_PATH)


if __name__ == "__main__":
    print("\n" + "─" * 50)
    print("  ML MODEL TESTS")
    print("─" * 50)
    test_linear_model()
    test_gbm_model()
    test_ensemble_prediction()
    test_feature_extractor()
    test_persistence()
    print("─" * 50)
    print("  ALL TESTS PASSED\n")
