"""
SQLite Persistence Layer – durable storage for all bot state.

Stores:
  - Trade history (every order placed, filled, cancelled)
  - Position snapshots (current exposure per market)
  - Signal history (every score computed, even if not traded)
  - Calibration data (predictions vs actual outcomes)
  - Whale activity log
  - Performance metrics (daily PnL, win rate, drawdown)

All queries use parameterized statements to prevent SQL injection.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Database:
    """SQLite database for persistent bot state."""

    def __init__(self, db_path: str = "data/bot.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Create all tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                -- Trade history
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    condition_id TEXT NOT NULL,
                    question TEXT,
                    side TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    order_id TEXT,
                    status TEXT NOT NULL DEFAULT 'FILLED',
                    mode TEXT NOT NULL DEFAULT 'paper',
                    signal_score REAL,
                    ai_divergence REAL,
                    fair_value REAL,
                    whale_score REAL,
                    time_score REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Position tracking
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    shares REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    exit_price REAL,
                    pnl REAL,
                    opened_at REAL NOT NULL,
                    closed_at REAL,
                    UNIQUE(condition_id, token_id, side)
                );

                -- Signal history (every evaluation, traded or not)
                CREATE TABLE IF NOT EXISTS signal_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    condition_id TEXT NOT NULL,
                    market_prob REAL,
                    ai_prob REAL,
                    ai_divergence REAL,
                    fair_value REAL,
                    whale_score REAL,
                    time_score REAL,
                    composite_score REAL NOT NULL,
                    triggered INTEGER NOT NULL DEFAULT 0,
                    direction TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Whale activity log
                CREATE TABLE IF NOT EXISTS whale_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    condition_id TEXT NOT NULL,
                    wallet TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    usd_value REAL NOT NULL,
                    token_id TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Calibration (predictions vs outcomes)
                CREATE TABLE IF NOT EXISTS calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_id TEXT NOT NULL UNIQUE,
                    predicted_prob REAL NOT NULL,
                    actual_outcome REAL,  -- NULL until resolved
                    brier_score REAL,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    predicted_at REAL NOT NULL,
                    resolved_at REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Daily performance summary
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    trades INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    total_pnl REAL NOT NULL DEFAULT 0,
                    max_drawdown REAL NOT NULL DEFAULT 0,
                    avg_score REAL NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Balances
                CREATE TABLE IF NOT EXISTS balance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    balance REAL NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'paper',
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_trades_cond ON trades(condition_id);
                CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
                CREATE INDEX IF NOT EXISTS idx_signal_cond ON signal_log(condition_id);
                CREATE INDEX IF NOT EXISTS idx_signal_ts ON signal_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_whale_cond ON whale_log(condition_id);
                CREATE INDEX IF NOT EXISTS idx_calibration_resolved ON calibration(resolved);
            """)

    # ---- TRADES ------------------------------------------------------------
    def insert_trade(self, trade: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO trades (
                    timestamp, condition_id, question, side, token_id,
                    price, size, cost_usd, order_id, status, mode,
                    signal_score, ai_divergence, fair_value, whale_score, time_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get("timestamp", time.time()),
                trade["condition_id"],
                trade.get("question", ""),
                trade["side"],
                trade["token_id"],
                trade["price"],
                trade["size"],
                trade["cost_usd"],
                trade.get("order_id", ""),
                trade.get("status", "FILLED"),
                trade.get("mode", "paper"),
                trade.get("signal_score"),
                trade.get("ai_divergence"),
                trade.get("fair_value"),
                trade.get("whale_score"),
                trade.get("time_score"),
            ))
            return cursor.lastrowid

    def get_trades(self, condition_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            if condition_id:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE condition_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (condition_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_trade_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buys,
                    SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sells,
                    COALESCE(SUM(cost_usd), 0) as total_volume,
                    COALESCE(AVG(signal_score), 0) as avg_score
                FROM trades
            """).fetchone()
            return dict(row)

    # ---- POSITIONS ---------------------------------------------------------
    def open_position(self, condition_id: str, token_id: str, side: str,
                      entry_price: float, shares: float, cost_usd: float) -> int:
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO positions (condition_id, token_id, side, entry_price, shares, cost_usd, opened_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (condition_id, token_id, side, entry_price, shares, cost_usd, time.time()))
            return cursor.lastrowid

    def close_position(self, condition_id: str, token_id: str, exit_price: float):
        with self._conn() as conn:
            conn.execute("""
                UPDATE positions
                SET status = 'CLOSED', exit_price = ?, closed_at = ?,
                    pnl = (exit_price - entry_price) * shares
                WHERE condition_id = ? AND token_id = ? AND status = 'OPEN'
            """, (exit_price, time.time(), condition_id, token_id))

    def get_open_positions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM positions WHERE status = 'OPEN'").fetchall()
            return [dict(r) for r in rows]

    def get_total_exposure(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) as total FROM positions WHERE status = 'OPEN'").fetchone()
            return row["total"]

    # ---- SIGNAL LOG --------------------------------------------------------
    def log_signal(self, condition_id: str, market_prob: float, ai_prob: float,
                   ai_divergence: float, fair_value: Optional[float],
                   whale_score: float, time_score: float,
                   composite_score: float, triggered: bool, direction: Optional[str]):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO signal_log (
                    timestamp, condition_id, market_prob, ai_prob, ai_divergence,
                    fair_value, whale_score, time_score, composite_score, triggered, direction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), condition_id, market_prob, ai_prob, ai_divergence,
                fair_value, whale_score, time_score, composite_score,
                1 if triggered else 0, direction,
            ))

    # ---- WHALE LOG ---------------------------------------------------------
    def log_whale(self, condition_id: str, wallet: str, side: str,
                  price: float, size: float, usd_value: float, token_id: str = ""):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO whale_log (timestamp, condition_id, wallet, side, price, size, usd_value, token_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (time.time(), condition_id, wallet, side, price, size, usd_value, token_id))

    def get_recent_whales(self, hours: float = 1.0) -> list[dict]:
        cutoff = time.time() - (hours * 3600)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM whale_log WHERE timestamp > ? ORDER BY timestamp DESC",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- CALIBRATION -------------------------------------------------------
    def record_prediction(self, condition_id: str, predicted_prob: float):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO calibration (condition_id, predicted_prob, predicted_at)
                VALUES (?, ?, ?)
            """, (condition_id, predicted_prob, time.time()))

    def resolve_prediction(self, condition_id: str, outcome: bool):
        actual = 1.0 if outcome else 0.0
        with self._conn() as conn:
            row = conn.execute(
                "SELECT predicted_prob FROM calibration WHERE condition_id = ?",
                (condition_id,),
            ).fetchone()
            if row:
                brier = (row["predicted_prob"] - actual) ** 2
                conn.execute("""
                    UPDATE calibration
                    SET actual_outcome = ?, brier_score = ?, resolved = 1, resolved_at = ?
                    WHERE condition_id = ?
                """, (actual, brier, time.time(), condition_id))

    def get_brier_score(self) -> Optional[float]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT AVG(brier_score) as avg_brier FROM calibration WHERE resolved = 1"
            ).fetchone()
            return row["avg_brier"]

    # ---- BALANCE LOG -------------------------------------------------------
    def log_balance(self, balance: float, mode: str = "paper"):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO balance_log (timestamp, balance, mode) VALUES (?, ?, ?)",
                (time.time(), balance, mode),
            )

    def get_balance_history(self, limit: int = 500) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM balance_log ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- ANALYTICS ---------------------------------------------------------
    def get_performance_summary(self) -> dict:
        """Generate a comprehensive performance report."""
        with self._conn() as conn:
            trades = conn.execute("SELECT COUNT(*) as total FROM trades").fetchone()
            open_pos = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) as exp FROM positions WHERE status = 'OPEN'").fetchone()
            last_balance = conn.execute("SELECT balance FROM balance_log ORDER BY timestamp DESC LIMIT 1").fetchone()
            brier = conn.execute("SELECT AVG(brier_score) as brier FROM calibration WHERE resolved = 1").fetchone()
            signals = conn.execute("SELECT COUNT(*) as total, SUM(triggered) as fired FROM signal_log").fetchone()

            return {
                "total_trades": trades["total"],
                "open_exposure": open_pos["exp"],
                "last_balance": last_balance["balance"] if last_balance else 0,
                "brier_score": brier["brier"],
                "total_signals": signals["total"],
                "signals_fired": signals["fired"],
            }
