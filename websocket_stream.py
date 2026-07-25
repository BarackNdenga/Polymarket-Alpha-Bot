"""
WebSocket Real-Time Stream – live market data without polling.

Connects to Polymarket's WebSocket API for:
  - Real-time order book updates (bids/asks changes)
  - Live trade feed (every trade as it happens)
  - Market state changes (price updates, end events)
  - Whale alerts (large trades detected in real-time)

This replaces the 60-second polling loop with sub-second updates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import websockets
import websockets.client

logger = logging.getLogger(__name__)

# Polymarket WebSocket endpoints
WS_MARKET_STREAM = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_TRADE_STREAM = "wss://ws-subscriptions-clob.polymarket.com/ws/trades"
WS_ACCOUNT_STREAM = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


@dataclass
class MarketEvent:
    """A real-time market event from the WebSocket stream."""
    event_type: str  # "price_update" | "book_update" | "trade" | "market_end"
    condition_id: str
    token_id: str
    data: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class TradeEvent:
    """A real-time trade event."""
    condition_id: str
    side: str  # "BUY" | "SELL"
    price: float
    size: float
    usd_value: float
    maker: str
    taker: str
    timestamp: float = field(default_factory=time.time)

    @property
    def is_whale(self) -> float:
        return self.usd_value >= 500.0


class WebSocketStream:
    """
    Real-time WebSocket client for Polymarket market data.

    Usage:
        stream = WebSocketStream()
        stream.on_trade(my_trade_handler)
        stream.on_price_update(my_price_handler)
        await stream.connect()
        await stream.subscribe_market("0xabc123")
    """

    def __init__(self):
        self._ws: Optional[websockets.client.WebSocketClientProtocol] = None
        self._running = False
        self._subscriptions: set[str] = set()
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._reconnect_delay = 5.0
        self._max_reconnect_delay = 120.0
        self._stats = {
            "events_received": 0,
            "trades_received": 0,
            "price_updates": 0,
            "reconnections": 0,
            "downtime_sec": 0.0,
        }

    # ---- Handler registration ----------------------------------------------
    def on_trade(self, callback: Callable[[TradeEvent], None]):
        self._handlers["trade"].append(callback)

    def on_price_update(self, callback: Callable[[MarketEvent], None]):
        self._handlers["price_update"].append(callback)

    def on_book_update(self, callback: Callable[[MarketEvent], None]):
        self._handlers["book_update"].append(callback)

    def on_whale(self, callback: Callable[[TradeEvent], None]):
        self._handlers["whale"].append(callback)

    def on_market_end(self, callback: Callable[[MarketEvent], None]):
        self._handlers["market_end"].append(callback)

    # ---- Connection --------------------------------------------------------
    async def connect(self):
        """Connect to the WebSocket and start listening."""
        self._running = True
        logger.info("WebSocket: connecting to Polymarket stream...")

        while self._running:
            try:
                self._ws = await websockets.connect(
                    WS_TRADE_STREAM,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                )
                logger.info("WebSocket: connected")
                self._reconnect_delay = 5.0

                # Re-subscribe after reconnection
                for cond_id in self._subscriptions:
                    await self._subscribe(cond_id)

                await self._listen()

            except (websockets.exceptions.ConnectionClosed, ConnectionError) as exc:
                self._stats["reconnections"] += 1
                self._stats["downtime_sec"] += self._reconnect_delay
                logger.warning("WebSocket: connection lost (%s). Reconnecting in %.0fs...", exc, self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 1.5, self._max_reconnect_delay)
            except Exception as exc:
                logger.error("WebSocket: unexpected error: %s", exc)
                await asyncio.sleep(self._reconnect_delay)

    async def disconnect(self):
        """Gracefully disconnect."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("WebSocket: disconnected")

    # ---- Subscription ------------------------------------------------------
    async def subscribe_market(self, condition_id: str):
        """Subscribe to a market's live stream."""
        self._subscriptions.add(condition_id)
        if self._ws:
            await self._subscribe(condition_id)
        logger.info("WebSocket: subscribed to %s", condition_id)

    async def unsubscribe_market(self, condition_id: str):
        """Unsubscribe from a market."""
        self._subscriptions.discard(condition_id)
        if self._ws:
            msg = json.dumps({"type": "unsubscribe", "markets": [condition_id]})
            await self._ws.send(msg)

    async def _subscribe(self, condition_id: str):
        """Send subscription message."""
        msg = json.dumps({
            "type": "subscribe",
            "markets": [condition_id],
        })
        await self._ws.send(msg)

    # ---- Listen loop -------------------------------------------------------
    async def _listen(self):
        """Main receive loop."""
        async for raw_msg in self._ws:
            if not self._running:
                break

            try:
                msg = json.loads(raw_msg)
                await self._dispatch(msg)
            except json.JSONDecodeError:
                continue
            except Exception as exc:
                logger.error("WebSocket dispatch error: %s", exc)

    async def _dispatch(self, msg: dict):
        """Route incoming messages to appropriate handlers."""
        msg_type = msg.get("type", "").lower()
        self._stats["events_received"] += 1

        if msg_type in ("trade", "fill", "match"):
            await self._handle_trade(msg)
        elif msg_type in ("price_update", "book", "book_update"):
            await self._handle_book(msg)
        elif msg_type == "market_end":
            await self._handle_market_end(msg)
        elif msg_type == "subscription_ack":
            logger.debug("WebSocket: subscription acknowledged")

    async def _handle_trade(self, msg: dict):
        """Process a trade event."""
        self._stats["trades_received"] += 1

        event = TradeEvent(
            condition_id=msg.get("market", ""),
            side=msg.get("side", "BUY"),
            price=float(msg.get("price", 0)),
            size=float(msg.get("size", 0)),
            usd_value=float(msg.get("price", 0)) * float(msg.get("size", 0)),
            maker=msg.get("maker", ""),
            taker=msg.get("taker", ""),
        )

        for cb in self._handlers.get("trade", []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as exc:
                logger.error("Trade handler error: %s", exc)

        # Whale alert
        if event.is_whale:
            for cb in self._handlers.get("whale", []):
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(event)
                    else:
                        cb(event)
                except Exception as exc:
                    logger.error("Whale handler error: %s", exc)

    async def _handle_book(self, msg: dict):
        """Process a book/price update event."""
        self._stats["price_updates"] += 1

        event = MarketEvent(
            event_type="price_update",
            condition_id=msg.get("market", ""),
            token_id=msg.get("token_id", ""),
            data=msg,
        )

        for cb in self._handlers.get("price_update", []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as exc:
                logger.error("Price handler error: %s", exc)

        for cb in self._handlers.get("book_update", []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as exc:
                logger.error("Book handler error: %s", exc)

    async def _handle_market_end(self, msg: dict):
        """Process market resolution/end event."""
        event = MarketEvent(
            event_type="market_end",
            condition_id=msg.get("market", ""),
            token_id=msg.get("token_id", ""),
            data=msg,
        )

        for cb in self._handlers.get("market_end", []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as exc:
                logger.error("Market end handler error: %s", exc)

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open
