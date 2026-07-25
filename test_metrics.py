"""
Unit tests for the Prometheus Metrics module.
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.metrics import MetricsState, METRICS, register_bot_metrics


def test_metrics_state():
    """Test thread-safe metrics holder."""
    ms = MetricsState()

    # Gauges
    ms.set_gauge("balance", 10000.0)
    ms.set_gauge("pnl", -4.92)
    all_metrics = ms.get_all()
    assert all_metrics["gauges"]["balance"] == 10000.0
    assert all_metrics["gauges"]["pnl"] == -4.92

    # Counters
    ms.inc_counter("trades", 1)
    ms.inc_counter("trades", 1)
    ms.inc_counter("signals", 5)
    all_metrics = ms.get_all()
    assert all_metrics["counters"]["trades"] == 2
    assert all_metrics["counters"]["signals"] == 5

    print("  MetricsState test PASSED")


def test_global_registration():
    """Test global metric registration."""
    register_bot_metrics()
    all_m = METRICS.get_all()
    assert "bot_balance_usd" in all_m["gauges"]
    assert "bot_trades_total" in all_m["counters"]
    assert "bot_composite_score" in all_m["gauges"]
    print("  Global registration test PASSED")


def test_metrics_update():
    """Test updating metrics."""
    METRICS.set_gauge("bot_balance_usd", 9995.08)
    METRICS.inc_counter("bot_trades_total", 1)

    all_m = METRICS.get_all()
    assert all_m["gauges"]["bot_balance_usd"] == 9995.08
    assert all_m["counters"]["bot_trades_total"] >= 1
    print("  Metrics update test PASSED")


if __name__ == "__main__":
    print("\n" + "─" * 50)
    print("  METRICS TESTS")
    print("─" * 50)
    test_metrics_state()
    test_global_registration()
    test_metrics_update()
    print("─" * 50)
    print("  ALL TESTS PASSED\n")
