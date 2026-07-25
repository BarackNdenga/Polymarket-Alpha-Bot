"""
Prometheus Metrics Exporter – expose bot metrics for monitoring.

Exports:
  - bot_balance_usd
  - bot_trades_total
  - bot_signals_fired_total
  - bot_markets_scanned_total
  - bot_whales_detected_total
  - bot_win_rate
  - bot_composite_score
  - bot_pnl_usd
  - bot_positions_open
  - bot_ws_events_total
  - bot_ws_connected (gauge)
  - bot_ml_accuracy (Brier score)
"""
from __future__ import annotations

import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsState:
    """Thread-safe metrics holder."""

    def __init__(self):
        self._lock = __import__("threading").Lock()
        self._gauges: dict[str, float] = {}
        self._counters: dict[str, float] = {}
        self._labels: dict[str, dict] = {}

    def set_gauge(self, name: str, value: float, labels: Optional[dict] = None):
        with self._lock:
            self._gauges[name] = value
            if labels:
                self._labels[name] = labels

    def inc_counter(self, name: str, value: float = 1.0):
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def get_all(self) -> dict:
        with self._lock:
            return {
                "gauges": dict(self._gauges),
                "counters": dict(self._counters),
            }


# Global metrics instance
METRICS = MetricsState()


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves Prometheus-format metrics."""

    def do_GET(self):
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return

        data = METRICS.get_all()
        lines = []

        # Gauges
        for name, value in data["gauges"].items():
            lines.append(f"# TYPE {name} gauge")
            labels = data.get("labels", {}).get(name, {})
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            if label_str:
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

        # Counters
        for name, value in data["counters"].items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}_total {value}")

        body = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        pass  # Suppress access logs


def start_metrics_server(port: int = 8080):
    """Start the metrics HTTP server in a background thread."""
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Metrics server started on port %d", port)
    return server


# ── Standard metric registration ──
def register_bot_metrics():
    """Register all standard bot metrics."""
    METRICS.set_gauge("bot_balance_usd", 0)
    METRICS.set_gauge("bot_pnl_usd", 0)
    METRICS.set_gauge("bot_positions_open", 0)
    METRICS.set_gauge("bot_composite_score", 0)
    METRICS.set_gauge("bot_win_rate", 0)
    METRICS.set_gauge("bot_ws_events_total", 0)
    METRICS.set_gauge("bot_ws_connected", 0)
    METRICS.set_gauge("bot_ml_accuracy", 0)
    METRICS.inc_counter("bot_trades_total", 0)
    METRICS.inc_counter("bot_signals_fired_total", 0)
    METRICS.inc_counter("bot_markets_scanned_total", 0)
    METRICS.inc_counter("bot_whales_detected_total", 0)
