"""
Unit tests for the Signal Engine.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signals.signal_engine import SignalEngine, SignalDirection


def test_strong_signal():
    """A strong signal should trigger a trade."""
    engine = SignalEngine()
    signal = engine.compute(
        condition_id="0xtest1",
        question="Will BTC reach $200k by Dec 31?",
        market_prob=0.30,
        ai_divergence=0.25,
        ai_confidence=0.80,
        fair_value=0.55,
        whale_score=0.70,
        whale_count=5,
        whale_direction="BUY",
        time_to_expiry_sec=7 * 86400,
    )
    assert signal.threshold_met, "Strong signal should trigger trade"
    assert signal.side == SignalDirection.BUY
    assert signal.composite_score > 0.65
    print(f"  Strong signal test PASSED (score={signal.composite_score:.4f})")


def test_weak_signal():
    """A weak signal should NOT trigger a trade."""
    engine = SignalEngine()
    signal = engine.compute(
        condition_id="0xtest2",
        question="Will it rain tomorrow?",
        market_prob=0.51,
        ai_divergence=0.02,
        ai_confidence=0.20,
        fair_value=None,
        whale_score=0.05,
        whale_count=0,
        whale_direction="NEUTRAL",
        time_to_expiry_sec=86400 * 30,
    )
    assert not signal.threshold_met, "Weak signal should NOT trigger trade"
    print(f"  Weak signal test PASSED (score={signal.composite_score:.4f})")


def test_time_decay_scoring():
    """Time decay should be highest for near-expiry markets."""
    engine = SignalEngine()

    t_1hr = engine.score_time_decay(3600)
    t_6hr = engine.score_time_decay(6 * 3600)
    t_1day = engine.score_time_decay(86400)
    t_7day = engine.score_time_decay(7 * 86400)

    assert t_1hr > t_6hr > t_1day, "Shorter expiry should score higher"
    print(f"  Time decay test PASSED (1h={t_1hr:.3f}, 6h={t_6hr:.3f}, 1d={t_1day:.3f}, 7d={t_7day:.3f})")


def test_fair_value_scoring():
    """Large fair value gap should produce high score."""
    engine = SignalEngine()
    s1 = engine.score_fair_value_gap(0.30, 0.60)  # 30% gap
    s2 = engine.score_fair_value_gap(0.45, 0.50)  # 5% gap
    assert s1 > s2, "Larger gap should score higher"
    print(f"  Fair value test PASSED (30% gap={s1:.3f}, 5% gap={s2:.3f})")


def test_ai_divergence_scoring():
    """AI divergence should be modulated by confidence."""
    engine = SignalEngine()
    s1 = engine.score_ai_divergence(0.20, 0.90)  # high divergence + high confidence
    s2 = engine.score_ai_divergence(0.20, 0.20)  # high divergence + low confidence
    assert s1 > s2, "High confidence should boost divergence score"
    print(f"  AI divergence test PASSED (high_conf={s1:.3f}, low_conf={s2:.3f})")


if __name__ == "__main__":
    print("\n" + "─" * 50)
    print("  SIGNAL ENGINE TESTS")
    print("─" * 50)
    test_strong_signal()
    test_weak_signal()
    test_time_decay_scoring()
    test_fair_value_scoring()
    test_ai_divergence_scoring()
    print("─" * 50)
    print("  ALL TESTS PASSED\n")
