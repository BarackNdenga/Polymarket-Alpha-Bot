"""
Global configuration for the Polymarket Alpha Bot.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV: str = os.getenv("BOT_ENV", "paper")  # "paper" | "live"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Polymarket credentials
# ---------------------------------------------------------------------------
POLYMARKET_PK: str = os.getenv("POLYMARKET_PK", "")  # L1 private key
POLYMARKET_API_KEY: str = os.getenv("POLYMARKET_API_KEY", "")  # L2 API key
POLYMARKET_API_SECRET: str = os.getenv("POLYMARKET_API_SECRET", "")
POLYMARKET_PASSPHRASE: str = os.getenv("POLYMARKET_PASSPHRASE", "")

# API base URLs
CLOB_URL: str = "https://clob.polymarket.com"
GAMMA_URL: str = "https://gamma-api.polymarket.com"
DATA_API_URL: str = "https://data-api.polymarket.com"

# ---------------------------------------------------------------------------
# PolymarketScan (AI vs Humans)
# ---------------------------------------------------------------------------
PMS_API_URL: str = "https://polymarketscan.org/api"
PMS_AI_CONSENSUS_URL: str = "https://api.polymarketscan.org/ai-consensus"

# ---------------------------------------------------------------------------
# Crypto price sources
# ---------------------------------------------------------------------------
BINANCE_TICKER_URL: str = "https://api.binance.com/api/v3/ticker/price"
COINGECKO_PRICE_URL: str = "https://api.coingecko.com/api/v3/simple/price"

# ---------------------------------------------------------------------------
# Whale tracking
# ---------------------------------------------------------------------------
WHALE_THRESHOLD_USD: float = 500.0  # minimum trade size to flag as whale
POLYWHALE_API: str = "https://polywhaler.com/api"

# ---------------------------------------------------------------------------
# Signal engine weights
# ---------------------------------------------------------------------------
WEIGHT_AI_DIVERGENCE: float = 0.35
WEIGHT_FAIR_VALUE_GAP: float = 0.30
WEIGHT_WHALE_MOMENTUM: float = 0.20
WEIGHT_TIME_DECAY: float = 0.15

# Minimum composite score to trigger a trade
SCORE_THRESHOLD: float = 0.65

# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------
MAX_POSITION_SIZE_USD: float = 5.0  # tiny live size for validation
PAPER_BALANCE: float = 10_000.0  # simulated balance
MAX_SPREAD_PCT: float = 0.04  # refuse orders wider than 4%
MIN_TIME_TO_EXPIRY_SEC: float = 3600.0  # 1h minimum to trade

# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
SCAN_INTERVAL_SEC: int = 60
TRADE_COOLDOWN_SEC: int = 300  # 5 min between trades on the same market


@dataclass
class ScoreWeights:
    """Weighted components of the composite alpha score."""
    ai_divergence: float = WEIGHT_AI_DIVERGENCE
    fair_value_gap: float = WEIGHT_FAIR_VALUE_GAP
    whale_momentum: float = WEIGHT_WHALE_MOMENTUM
    time_decay: float = WEIGHT_TIME_DECAY


WEIGHTS = ScoreWeights()
