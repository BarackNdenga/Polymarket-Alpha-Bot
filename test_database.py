"""
Unit tests for the SQLite Database layer.
"""
import sys
import os
import time
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.database import Database


def test_insert_trade():
    """Test trade insertion."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        trade_id = db.insert_trade({
            "timestamp": time.time(),
            "condition_id": "0xabc123",
            "question": "Will BTC hit $200k?",
            "side": "BUY",
            "token_id": "token_yes",
            "price": 0.35,
            "size": 14,
            "cost_usd": 4.90,
            "order_id": "order_001",
            "status": "FILLED",
            "mode": "paper",
            "signal_score": 0.72,
            "ai_divergence": 0.22,
            "fair_value": 0.52,
            "whale_score": 0.65,
            "time_score": 0.15,
        })
        assert trade_id is not None

        trades = db.get_trades()
        assert len(trades) == 1
        assert trades[0]["condition_id"] == "0xabc123"
        assert trades[0]["side"] == "BUY"
        print("  Insert trade test PASSED")

        # Filter by condition
        trades_filtered = db.get_trades(condition_id="0xabc123")
        assert len(trades_filtered) == 1
        trades_empty = db.get_trades(condition_id="0xnotexist")
        assert len(trades_empty) == 0
        print("  Filter by condition PASSED")
    finally:
        os.unlink(db_path)


def test_positions():
    """Test position tracking."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)

        # Open position
        pos_id = db.open_position(
            condition_id="0xabc",
            token_id="token_yes",
            side="BUY",
            entry_price=0.35,
            shares=14,
            cost_usd=4.90,
        )
        assert pos_id is not None

        # Check open positions
        open_pos = db.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0]["side"] == "BUY"

        # Check exposure
        exposure = db.get_total_exposure()
        assert exposure == 4.90
        print("  Position tracking test PASSED")

        # Close position
        db.close_position("0xabc", "token_yes", 0.42)
        open_pos = db.get_open_positions()
        assert len(open_pos) == 0
        print("  Close position PASSED")
    finally:
        os.unlink(db_path)


def test_signal_log():
    """Test signal logging."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        db.log_signal(
            condition_id="0xabc",
            market_prob=0.35,
            ai_prob=0.57,
            ai_divergence=0.22,
            fair_value=0.52,
            whale_score=0.65,
            time_score=0.15,
            composite_score=0.72,
            triggered=True,
            direction="BUY",
        )
        # Should not raise
        print("  Signal log test PASSED")
    finally:
        os.unlink(db_path)


def test_whale_log():
    """Test whale logging."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        db.log_whale("0xabc", "0x742d", "BUY", 0.35, 1000, 350.0, "token_yes")
        db.log_whale("0xdef", "0xa91c", "SELL", 0.62, 500, 310.0, "token_no")

        whales = db.get_recent_whales(hours=1.0)
        assert len(whales) == 2
        assert whales[0]["side"] == "SELL"  # Most recent first

        whales_old = db.get_recent_whales(hours=-1.0)
        assert len(whales_old) == 0
        print("  Whale log test PASSED")
    finally:
        os.unlink(db_path)


def test_balance_log():
    """Test balance logging."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        db.log_balance(10000.0, "paper")
        db.log_balance(9995.08, "paper")
        db.log_balance(10012.50, "paper")

        history = db.get_balance_history()
        assert len(history) == 3
        assert history[0]["balance"] == 10012.50  # Most recent
        print("  Balance log test PASSED")
    finally:
        os.unlink(db_path)


def test_calibration():
    """Test prediction calibration."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        db.record_prediction("0xabc", 0.65)
        db.record_prediction("0xdef", 0.30)

        db.resolve_prediction("0xabc", True)   # Predicted 0.65, actual 1.0
        db.resolve_prediction("0xdef", False)   # Predicted 0.30, actual 0.0

        brier = db.get_brier_score()
        assert brier is not None
        # Brier = avg of (0.65-1)^2, (0.30-0)^2 = avg(0.1225, 0.09) = 0.10625
        assert abs(brier - 0.10625) < 0.001
        print(f"  Calibration test PASSED (Brier={brier:.5f})")
    finally:
        os.unlink(db_path)


def test_performance_summary():
    """Test performance summary generation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        db.insert_trade({
            "timestamp": time.time(),
            "condition_id": "0xabc",
            "question": "Test",
            "side": "BUY",
            "token_id": "token_yes",
            "price": 0.35,
            "size": 14,
            "cost_usd": 4.90,
            "mode": "paper",
        })
        db.log_signal("0xabc", 0.35, 0.57, 0.22, 0.52, 0.65, 0.15, 0.72, True, "BUY")
        db.log_balance(9995.08, "paper")

        summary = db.get_performance_summary()
        assert summary["total_trades"] == 1
        assert summary["signals_fired"] == 1
        assert summary["last_balance"] == 9995.08
        print("  Performance summary test PASSED")
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    print("\n" + "─" * 50)
    print("  DATABASE TESTS")
    print("─" * 50)
    test_insert_trade()
    test_positions()
    test_signal_log()
    test_whale_log()
    test_balance_log()
    test_calibration()
    test_performance_summary()
    print("─" * 50)
    print("  ALL TESTS PASSED\n")
