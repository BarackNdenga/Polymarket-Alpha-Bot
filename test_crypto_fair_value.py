"""
Unit tests for the Crypto Fair Value module.
"""
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.crypto_prices import CryptoFairValue, CryptoPriceFetcher


def test_price_target_probability():
    """Test the Black-Scholes-style probability calculation."""
    # BTC at $100k, target $120k, 30 days, 75% IV
    prob = CryptoFairValue.price_target_probability(
        current_price=100_000,
        target_price=120_000,
        days_remaining=30,
        implied_volatility=0.75,
    )
    assert 0 < prob < 1, f"Probability should be between 0 and 1, got {prob}"
    # Higher target should give lower probability
    prob2 = CryptoFairValue.price_target_probability(
        current_price=100_000,
        target_price=200_000,
        days_remaining=30,
        implied_volatility=0.75,
    )
    assert prob > prob2, f"Lower target should have higher probability ({prob} vs {prob2})"
    print(f"  Price target prob (100k→120k, 30d): {prob:.4f}")
    print(f"  Price target prob (100k→200k, 30d): {prob2:.4f}")


def test_longer_time_higher_prob():
    """More time should increase probability of reaching target."""
    prob_7d = CryptoFairValue.price_target_probability(100_000, 110_000, 7)
    prob_30d = CryptoFairValue.price_target_probability(100_000, 110_000, 30)
    prob_90d = CryptoFairValue.price_target_probability(100_000, 110_000, 90)
    assert prob_7d < prob_30d < prob_90d, "Longer time → higher probability"
    print(f"  Time effect: 7d={prob_7d:.4f}, 30d={prob_30d:.4f}, 90d={prob_90d:.4f}")


def test_question_extraction():
    """Test parsing target prices from market questions."""
    tests = [
        ("Will Bitcoin reach $150,000 by December 2026?", ("btc", 150000)),
        ("Will ETH hit $10,000 this year?", ("eth", 10000)),
        ("Will Solana exceed $500 by Q3?", ("sol", 500)),
    ]
    for question, expected in tests:
        result = CryptoFairValue.extract_target_from_question(question)
        assert result is not None, f"Failed to parse: {question}"
        assert result[0] == expected[0], f"Wrong asset: {result[0]} vs {expected[0]}"
        assert result[1] == expected[1], f"Wrong price: {result[1]} vs {expected[1]}"
        print(f"  Extracted from '{question[:40]}...': {result}")


def test_invalid_inputs():
    """Edge cases should return 0."""
    assert CryptoFairValue.price_target_probability(0, 100, 30) == 0.0
    assert CryptoFairValue.price_target_probability(100, 0, 30) == 0.0
    assert CryptoFairValue.price_target_probability(100, 200, 0) == 0.0
    print(f"  Invalid input edge cases PASSED")


if __name__ == "__main__":
    print("\n" + "─" * 50)
    print("  CRYPTO FAIR VALUE TESTS")
    print("─" * 50)
    test_price_target_probability()
    test_longer_time_higher_prob()
    test_question_extraction()
    test_invalid_inputs()
    print("─" * 50)
    print("  ALL TESTS PASSED\n")
