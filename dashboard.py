"""
Dashboard & Analytics – real-time monitoring and trade analytics.

Provides:
  - Real-time signal dashboard (CLI-based)
  - Trade analytics and PnL tracking
  - Export to JSON/CSV for external analysis
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import ENV


@dataclass
class DashboardState:
    """Aggregated state for the dashboard display."""
    total_scans: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_score: float = 0.0
    markets_evaluated: int = 0
    signals_triggered: int = 0
    buy_orders: int = 0
    sell_orders: int = 0
    last_scan_time: str = ""
    mode: str = ENV


class Dashboard:
    """CLI dashboard and analytics exporter."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.state = DashboardState()

    def render(self) -> str:
        """Render the dashboard as a formatted string."""
        border = "═" * 62
        thin = "─" * 62

        return f"""
┌{border}┐
│{' POLYMARKET ALPHA BOT — DASHBOARD':^60}│
├{thin}┤
│ Mode:           {self.state.mode:<43} │
│ Scans:          {self.state.total_scans:<43} │
│ Markets Eval:   {self.state.markets_evaluated:<43} │
│ Signals Fired:  {self.state.signals_triggered:<43} │
│ Trades Placed:  {self.state.total_trades:<43} │
│   Buy Orders:   {self.state.buy_orders:<43} │
│   Sell Orders:  {self.state.sell_orders:<43} │
├{thin}┤
│ Avg Score:      {self.state.avg_score:<43.4f} │
│ Win Rate:       {self.state.win_rate:<43.2%} │
│ Total PnL:      ${self.state.total_pnl:<42.2f} │
├{thin}┤
│ Last Scan:      {self.state.last_scan_time:<43} │
└{border}┘
"""

    def load_from_trades(self, trades_file: Optional[Path] = None):
        """Load trade data from JSONL and update dashboard state."""
        if trades_file is None:
            trades_file = self.log_dir / f"trades_{ENV}.jsonl"

        if not trades_file.exists():
            return

        trades = []
        with open(trades_file) as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))

        if not trades:
            return

        self.state.total_trades = len(trades)
        self.state.buy_orders = sum(1 for t in trades if t.get("side") == "BUY")
        self.state.sell_orders = sum(1 for t in trades if t.get("side") == "SELL")

        scores = [t.get("score", 0) for t in trades]
        self.state.avg_score = sum(scores) / len(scores) if scores else 0

        self.state.last_scan_time = trades[-1].get("timestamp", "")

    def export_json(self, filename: str = "analytics_export.json"):
        """Export dashboard state and trade history to JSON."""
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "mode": ENV,
            "state": {
                "total_scans": self.state.total_scans,
                "total_trades": self.state.total_trades,
                "total_pnl": self.state.total_pnl,
                "win_rate": self.state.win_rate,
                "avg_score": self.state.avg_score,
                "markets_evaluated": self.state.markets_evaluated,
                "signals_triggered": self.state.signals_triggered,
                "buy_orders": self.state.buy_orders,
                "sell_orders": self.state.sell_orders,
            },
        }

        # Load trades
        trades_file = self.log_dir / f"trades_{ENV}.jsonl"
        if trades_file.exists():
            trades = []
            with open(trades_file) as f:
                for line in f:
                    if line.strip():
                        trades.append(json.loads(line))
            data["trades"] = trades

        out = Path(filename)
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported analytics to {out}")

    def export_csv(self, filename: str = "trades_export.csv"):
        """Export trade log to CSV."""
        import csv

        trades_file = self.log_dir / f"trades_{ENV}.jsonl"
        if not trades_file.exists():
            print("No trades to export.")
            return

        with open(trades_file) as fin, open(filename, "w", newline="") as fout:
            lines = [json.loads(l) for l in fin if l.strip()]
            if not lines:
                return
            writer = csv.DictWriter(fout, fieldnames=lines[0].keys())
            writer.writeheader()
            writer.writerows(lines)
        print(f"Exported {len(lines)} trades to {filename}")
