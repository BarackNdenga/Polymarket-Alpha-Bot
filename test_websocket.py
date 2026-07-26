"""
Unit tests for the WebSocket Stream module.
"""
import sys
import os
import asyncio
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.websocket_stream import WebSocketStream, MarketEvent, TradeEvent


def test_trade_event():
    """Test TradeEvent creation and whale detection."""
    event = TradeEvent(
        condition_id="0xabc123",
        side="BUY",
        price=0.45,
        size=1200,
        usd_value=540.0,
        maker="0x742d...8f3a",
        taker="0xa91c...2e4b",
    )
    assert event.is_whale == True, "540 USD should be flagged as whale"
    assert event.side == "BUY"
    assert event.price == 0.45
    print("  TradeEvent test PASSED")

    # Non-whale
    small = TradeEvent(
        condition_id="0xdef456",
        side="SELL",
        price=0.30,
        size=10,
        usd_value=3.0,
        maker="0x1234",
        taker="0x5678",
    )
    assert small.is_whale == False, "3 USD should NOT be whale"
    print("  Non-whale detection PASSED")


def test_market_event():
    """Test MarketEvent creation."""
    event = MarketEvent(
        event_type="price_update",
        condition_id="0xabc",
        token_id="token1",
        data={"mid": 0.45},
    )
    assert event.event_type == "price_update"
    assert event.condition_id == "0xabc"
    assert event.timestamp > 0
    print("  MarketEvent test PASSED")


def test_stream_handler_registration():
    """Test that handlers can be registered."""
    stream = WebSocketStream()
    called = []

    def my_handler(event):
        called.append(event)

    stream.on_trade(my_handler)
    stream.on_price_update(my_handler)
    stream.on_whale(my_handler)
    stream.on_book_update(my_handler)
    stream.on_market_end(my_handler)

    assert len(stream._handlers["trade"]) == 1
    assert len(stream._handlers["price_update"]) == 1
    assert len(stream._handlers["whale"]) == 1
    print("  Handler registration test PASSED")


def test_stream_stats():
    """Test stats tracking."""
    stream = WebSocketStream()
    stats = stream.stats
    assert "events_received" in stats
    assert "trades_received" in stats
    assert "price_updates" in stats
    assert "reconnections" in stats
    assert stats["events_received"] == 0
    print("  Stats test PASSED")


def test_stream_connection_state():
    """Test connection state."""
    stream = WebSocketStream()
    assert stream.is_connected == False
    print("  Connection state test PASSED")


if __name__ == "__main__":
    print("\n" + "─" * 50)
    print("  WEBSOCKET TESTS")
    print("─" * 50)
    test_trade_event()
    test_market_event()
    test_stream_handler_registration()
    test_stream_stats()
    test_stream_connection_state()
    print("─" * 50)
    print("  ALL TESTS PASSED\n")
