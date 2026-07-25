"""
Core Orchestrator – the main loop that ties everything together.

Full integration:
  - WebSocket real-time streaming
  - SQLite persistence
  - ML model predictions
  - Prometheus metrics
  - Risk management
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import (
    ENV,
    MIN_TIME_TO_EXPIRY_SEC,
    SCAN_INTERVAL_SEC,
)
from src.data.ai_consensus import AIConsensusFetcher
from src.data.crypto_prices import CryptoFairValue, CryptoPriceFetcher
from src.data.database import Database
from src.data.polymarket_data import PolymarketDataFetcher, MarketBook
from src.data.websocket_stream import WebSocketStream, TradeEvent
from src.data.whale_tracker import WhaleTracker
from src.execution.order_executor import (
    ExecutionMode,
    OrderExecutor,
    OrderSide,
    PaperTradingEngine,
    LiveOrderExecutor,
)
from src.signals.ml_model import FeatureExtractor, MLSignalModel
from src.signals.signal_engine import SignalEngine
from src.utils.risk_manager import RiskManager
from src.core.metrics import METRICS, register_bot_metrics, start_metrics_server

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main bot loop – scans markets, computes signals, places orders."""

    def __init__(self):
        # Data sources
        self.pm = PolymarketDataFetcher()
        self.ai = AIConsensusFetcher()
        self.crypto_prices = CryptoPriceFetcher()
        self.crypto_fv = CryptoFairValue(self.crypto_prices)
        self.whales = WhaleTracker()

        # WebSocket stream
        self.ws = WebSocketStream()

        # Database
        self.db = Database()

        # ML Model
        self.ml = MLSignalModel()
        self.ml.load()
        self.ml.load_training_data()

        # Signal engine
        self.signals = SignalEngine()

        # Feature extractor
        self.features = FeatureExtractor()

        # Risk manager
        self.risk = RiskManager()

        # Execution
        mode = ExecutionMode.PAPER if ENV == "paper" else ExecutionMode.LIVE
        self.executor = OrderExecutor(mode=mode)

        # Logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Stats
        self.scan_count = 0
        self.trade_count = 0
        self.start_time = time.time()

        # Register metrics
        register_bot_metrics()

    # ---- WebSocket handlers ------------------------------------------------
    def setup_websocket_handlers(self):
        """Register real-time WebSocket callbacks."""

        async def on_trade(event: TradeEvent):
            METRICS.set_gauge("bot_ws_events_total", self.ws.stats["trades_received"])
            # Log whale trades
            if event.is_whale:
                self.db.log_whale(
                    condition_id=event.condition_id,
                    wallet=event.maker,
                    side=event.side,
                    price=event.price,
                    size=event.size,
                    usd_value=event.usd_value,
                )
                METRICS.inc_counter("bot_whales_detected_total")
                logger.info("WHALE: %s %s $%.0f on %s", event.side, event.size, event.usd_value, event.condition_id[:12])

        async def on_price(event):
            METRICS.set_gauge("bot_ws_events_total", self.ws.stats["price_updates"])

        async def on_whale(event: TradeEvent):
            logger.info("🐋 WHALE ALERT: %s %s shares @ %.4f ($%.0f) — %s",
                        event.side, event.size, event.price, event.usd_value, event.condition_id[:16])

        self.ws.on_trade(on_trade)
        self.ws.on_price_update(on_price)
        self.ws.on_whale(on_whale)

    # ---- Single market evaluation ------------------------------------------
    def evaluate_market(self, market: dict) -> bool:
        """
        Evaluate a single market. Returns True if a trade was placed.
        """
        # Build MarketBook
        book = self.pm.build_market_book(market)
        if book is None:
            return False

        condition_id = book.condition_id
        question = book.question

        # Skip inactive or soon-to-expire markets
        if not book.active:
            return False

        tte = book.time_to_expiry()
        if tte < MIN_TIME_TO_EXPIRY_SEC:
            return False

        # 1. AI Consensus
        consensus = self.ai.get_consensus(
            condition_id=condition_id,
            human_midpoint=book.midpoint,
            question=question,
            volume_24h=book.volume_24h,
            spread=book.spread,
        )

        # 2. Fair Value (crypto markets only)
        fair_value: Optional[float] = None
        if any(kw in question.lower() for kw in ["btc", "bitcoin", "eth", "ethereum", "sol", "solana", "crypto"]):
            fair_value = self.crypto_fv.auto_fair_value(question, tte / 86400.0)

        # 3. Whale Momentum
        whale_sig = self.whales.compute_signal(condition_id)

        # 4. Classic Signal Engine
        classic_signal = self.signals.compute(
            condition_id=condition_id,
            question=question,
            market_prob=book.midpoint,
            ai_divergence=consensus.divergence,
            ai_confidence=consensus.confidence,
            fair_value=fair_value,
            whale_score=whale_sig.score,
            whale_count=whale_sig.whale_count,
            whale_direction=whale_sig.dominant_side,
            time_to_expiry_sec=tte,
        )

        # 5. ML Model prediction
        feat = self.features.extract(
            market=market,
            book=self.pm.order_book(book.yes_token_id),
            ai_divergence=consensus.divergence,
            ai_confidence=consensus.confidence,
            fair_value=fair_value,
            whale_momentum=whale_sig.score,
            whale_count=whale_sig.whale_count,
            whale_net_flow=whale_sig.net_flow_usd,
            time_to_expiry_hours=tte / 3600,
            question=question,
            volume_24h=book.volume_24h,
        )
        ml_prediction = self.ml.predict_mispricing(feat)

        # 6. Combined decision
        # Use ML if trained, otherwise fall back to classic
        if self.ml.gbm_model.is_fitted and ml_prediction["is_mispriced"]:
            # ML says mispriced → boost or override classic signal
            if ml_prediction["direction"] == classic_signal.side.value:
                # Both agree → extra confidence
                classic_signal.composite_score = min(1.0, classic_signal.composite_score * 1.2)
            elif ml_prediction["direction"] != classic_signal.side.value:
                # Disagreement → reduce confidence
                classic_signal.composite_score *= 0.7

        # Log signal
        self.db.log_signal(
            condition_id=condition_id,
            market_prob=book.midpoint,
            ai_prob=consensus.ai_probability,
            ai_divergence=consensus.divergence,
            fair_value=fair_value,
            whale_score=whale_sig.score,
            time_score=classic_signal.time_decay_score,
            composite_score=classic_signal.composite_score,
            triggered=classic_signal.threshold_met,
            direction=classic_signal.side.value,
        )

        # Update metrics
        METRICS.set_gauge("bot_composite_score", classic_signal.composite_score)
        METRICS.inc_counter("bot_markets_scanned_total")
        if classic_signal.threshold_met:
            METRICS.inc_counter("bot_signals_fired_total")

        logger.info(classic_signal.summary())

        # 7. Execute if threshold met + risk check
        if classic_signal.threshold_met:
            token_id = book.yes_token_id if classic_signal.side == OrderSide.BUY else book.no_token_id
            cost_usd = min(5.0, book.midpoint * 10)  # estimate

            approved, reason = self.risk.pre_trade_check(condition_id, classic_signal.side.value, cost_usd)
            if not approved:
                logger.info("Risk check failed for %s: %s", condition_id[:12], reason)
                return False

            result = self.executor.place_order(
                token_id=token_id,
                side=classic_signal.side,
                best_bid=book.yes_best_bid if classic_signal.side == OrderSide.BUY else book.no_best_bid,
                best_ask=book.yes_best_ask if classic_signal.side == OrderSide.BUY else book.no_best_ask,
                signal_score=classic_signal.composite_score,
                time_to_expiry_sec=tte,
                current_spread=book.spread,
                condition_id=condition_id,
            )
            if result is not None:
                self.trade_count += 1
                self.risk.post_trade_update(cost_usd)

                # Persist trade
                self.db.insert_trade({
                    "timestamp": time.time(),
                    "condition_id": condition_id,
                    "question": question,
                    "side": classic_signal.side.value,
                    "token_id": token_id,
                    "price": result.price,
                    "size": result.size,
                    "cost_usd": result.cost_usd,
                    "order_id": result.order_id,
                    "status": result.status.value,
                    "mode": result.mode.value,
                    "signal_score": classic_signal.composite_score,
                    "ai_divergence": classic_signal.ai_divergence_score,
                    "fair_value": fair_value,
                    "whale_score": classic_signal.whale_momentum_score,
                    "time_score": classic_signal.time_decay_score,
                })

                # Open position
                self.db.open_position(
                    condition_id=condition_id,
                    token_id=token_id,
                    side=classic_signal.side.value,
                    entry_price=result.price,
                    shares=result.size,
                    cost_usd=result.cost_usd,
                )

                self._log_trade(classic_signal, result, book)

                # Update balance metrics
                if self.executor.mode == ExecutionMode.PAPER:
                    METRICS.set_gauge("bot_balance_usd", self.executor.paper.balance)

                METRICS.inc_counter("bot_trades_total")
                return True

        return False

    # ---- Main scan loop ----------------------------------------------------
    def run(self, num_markets: int = 20, iterations: int = 10):
        """
        Run the scanning loop for a fixed number of iterations.
        """
        logger.info("=" * 60)
        logger.info("  POLYMARKET ALPHA BOT STARTING")
        logger.info(f"  Mode:        {ENV}")
        logger.info(f"  Scan interval: {SCAN_INTERVAL_SEC}s")
        logger.info(f"  Markets/scan:  {num_markets}")
        logger.info(f"  Iterations:    {iterations}")
        logger.info(f"  WebSocket:     enabled")
        logger.info(f"  Database:      {self.db.db_path}")
        logger.info(f"  ML Model:      {'loaded' if self.ml.gbm_model.is_fitted else 'training mode'}")
        logger.info("=" * 60)

        # Start metrics server
        start_metrics_server()

        # Setup WebSocket
        self.setup_websocket_handlers()

        # Start WebSocket in background
        ws_thread = asyncio.new_event_loop()
        import threading
        def run_ws():
            asyncio.set_event_loop(ws_thread)
            ws_thread.run_until_complete(self.ws.connect())
        t = threading.Thread(target=run_ws, daemon=True)
        t.start()

        for iteration in range(iterations):
            self.scan_count += 1
            logger.info(f"\n{'─' * 60}")
            logger.info(f"  SCAN #{self.scan_count} — {datetime.utcnow().isoformat()}Z")
            logger.info(f"{'─' * 60}")

            markets = self.pm.discover_markets(limit=num_markets)
            trades_this_scan = 0

            for market in markets:
                try:
                    traded = self.evaluate_market(market)
                    if traded:
                        trades_this_scan += 1
                except Exception as exc:
                    logger.error("Error evaluating market: %s", exc)

            logger.info(f"  Scan complete: {len(markets)} markets, {trades_this_scan} trades")

            # Update metrics
            METRICS.set_gauge("bot_positions_open", self.db.get_total_exposure())
            METRICS.set_gauge("bot_win_rate", self.trade_count / max(self.scan_count, 1))
            if self.executor.mode == ExecutionMode.PAPER:
                METRICS.set_gauge("bot_balance_usd", self.executor.paper.balance)
                logger.info(self.executor.performance_report())

            # Retrain ML every 10 scans if enough data
            if self.scan_count % 10 == 0 and len(self.ml._training_samples) >= 50:
                logger.info("Retraining ML model with %d samples...", len(self.ml._training_samples))
                self.ml.train()

            # Wait before next scan
            if iteration < iterations - 1:
                logger.info(f"  Waiting {SCAN_INTERVAL_SEC}s for next scan...")
                time.sleep(SCAN_INTERVAL_SEC)

        # Final report
        self._final_report()

    # ---- Logging -----------------------------------------------------------
    def _log_trade(self, signal, result, book: MarketBook):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "condition_id": signal.condition_id,
            "question": signal.question,
            "side": signal.side.value,
            "score": signal.composite_score,
            "market_prob": book.midpoint,
            "order_price": result.price,
            "order_size": result.size,
            "order_id": result.order_id,
            "mode": result.mode.value,
            "ai_divergence": signal.ai_divergence_score,
            "fair_value_score": signal.fair_value_score,
            "whale_score": signal.whale_momentum_score,
            "time_score": signal.time_decay_score,
        }
        log_file = self.log_dir / f"trades_{ENV}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _final_report(self):
        elapsed = time.time() - self.start_time
        perf = self.db.get_performance_summary()
        brier = self.db.get_brier_score()

        logger.info("\n" + "=" * 60)
        logger.info("  FINAL REPORT")
        logger.info(f"  Runtime:     {elapsed:.0f}s ({elapsed/60:.1f} min)")
        logger.info(f"  Scans:       {self.scan_count}")
        logger.info(f"  Trades:      {self.trade_count}")
        logger.info(f"  Signals:     {perf.get('signals_fired', 0)} fired / {perf.get('total_signals', 0)} total")
        logger.info(f"  Exposure:    ${perf.get('open_exposure', 0):.2f}")
        logger.info(f"  Brier Score: {brier or 'N/A'}")
        logger.info(f"  ML Status:   {'trained' if self.ml.gbm_model.is_fitted else 'collecting data'}")
        logger.info(self.executor.performance_report())
        logger.info(self.risk.status())
        logger.info("=" * 60)
