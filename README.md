# Polymarket Alpha Bot

**AI vs Humans + Cross-Platform Fair Value + Whale Shadowing + ML + Real-Time**

An autonomous trading agent that identifies mispricings in Polymarket prediction markets by comparing human market odds against AI consensus, external crypto fair values, and whale momentum. The bot executes intelligent limit orders with dynamic spread management.

## Architecture

The bot is built around a core orchestrator that evaluates markets through a multi-signal composite scoring engine:

1. **AI vs Humans**: Compares the human-derived market midpoint with a consensus of AI agents (fetched via PolymarketScan API or a local heuristic ensemble fallback).
2. **Cross-Platform Fair Value**: For crypto-related markets, calculates a model-implied probability using spot prices from Binance and CoinGecko.
3. **Whale Shadowing**: Monitors the Data API for large trades (>$500) to detect directional momentum.
4. **Real-Time Streaming**: Connects to Polymarket WebSockets to receive trade and order book updates instantly.
5. **Machine Learning**: Uses an ensemble of Ridge Regression and Gradient Boosting to predict mispricings based on engineered features.
6. **Time Decay**: Scores urgency based on time-to-expiry.

When the composite score exceeds a threshold, the bot places a limit order using a dynamic spread calculator that adjusts aggressiveness based on signal strength and market liquidity.

## Features

- **Smart Limit Orders**: Never places dumb market orders. Uses a dynamic spread calculator to place bids/asks optimally.
- **Paper Trading Mode**: Default mode uses a simulated balance ($10,000) to validate the strategy without risk.
- **Live Trading Mode**: Supports tiny-size (1-5 USDC) live trading for real-world validation.
- **Real-Time WebSocket**: Sub-second updates on trades, book changes, and whale alerts.
- **SQLite Persistence**: Durable storage for trade history, positions, and calibration data.
- **ML Model**: Predictive model with 19 engineered features, trained on historical outcomes.
- **Prometheus Metrics**: Exposes performance metrics on port 8080 for monitoring.
- **Risk Management**: Includes position sizing limits, daily loss limits, and circuit breakers.
- **Docker Deployment**: Production-ready Dockerfile and docker-compose setup.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/polymarket-alpha-bot.git
cd polymarket-alpha-bot

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional for paper trading)
export BOT_ENV=paper
export POLYMARKET_PK="your_private_key"  # Required for live trading
export POLYMARKET_API_KEY="your_api_key"
export POLYMARKET_API_SECRET="your_api_secret"
```

## Docker Deployment

For production, use the included Docker configuration:

```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f bot
```

The stack includes:
- `bot`: The main trading bot (restarts automatically)
- `ml-worker`: Optional worker that retrains the ML model hourly
- `dashboard`: Nginx server serving the HTML dashboard on port 8080

## Usage

### Demo Mode
Run a demo with synthetic data (no API calls required):
```bash
python run.py --demo
```

### Paper Trading (Default)
Scan the top 20 markets for 10 iterations:
```bash
python run.py --markets 20 --iterations 10
```

### Live Trading
Switch to live mode with tiny position sizes:
```bash
BOT_ENV=live python run.py
```

### Single Market Evaluation
Evaluate a specific market by its condition ID:
```bash
python run.py --market 0x1234567890abcdef
```

## Configuration

Edit `config/settings.py` to adjust the bot's behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WEIGHT_AI_DIVERGENCE` | 0.35 | Weight of AI vs Human signal |
| `WEIGHT_FAIR_VALUE_GAP` | 0.30 | Weight of cross-platform fair value |
| `WEIGHT_WHALE_MOMENTUM` | 0.20 | Weight of whale tracking |
| `WEIGHT_TIME_DECAY` | 0.15 | Weight of time-to-expiry urgency |
| `SCORE_THRESHOLD` | 0.65 | Minimum composite score to trigger a trade |
| `MAX_POSITION_SIZE_USD` | 5.0 | Maximum order size in USD |
| `MAX_SPREAD_PCT` | 0.04 | Refuse orders with spread wider than 4% |

## Testing

Run the full test suite:
```bash
bash tests/run_all.sh
```

Tests cover:
- Signal engine scoring logic
- Order executor spread calculations
- Crypto fair value model
- WebSocket event handling
- Database persistence
- ML model training and prediction
- Prometheus metrics

## Contribution Guidelines

We welcome contributions! Please follow these guidelines when submitting pull requests:

1. **Fork the Repository**: Create a fork of the project on GitHub.
2. **Create a Branch**: Create a feature branch (`git checkout -b feature/your-feature`).
3. **Code Style**: Follow PEP 8 standards. Use type hints and docstrings for all functions and classes.
4. **Testing**: Write unit tests for any new features or bug fixes in the `tests/` directory. Ensure all tests pass (`bash tests/run_all.sh`).
5. **Documentation**: Update the `README.md` if your changes affect usage or configuration.
6. **Commit**: Commit your changes (`git commit -m 'Add some feature'`).
7. **Push**: Push to the branch (`git push origin feature/your-feature`).
8. **Pull Request**: Open a Pull Request against the `main` branch of the upstream repository.

### Areas for Contribution

- **New Data Sources**: Integrating additional prediction market platforms (e.g., Kalshi, Manifold).
- **Advanced Models**: Improving the crypto fair value model with more sophisticated volatility estimators.
- **UI/UX**: Building a web-based dashboard to replace the CLI dashboard.
- **Performance**: Optimizing API calls and data fetching for lower latency.

## Disclaimer

This software is for educational and research purposes only. Prediction market trading involves financial risk. The authors are not responsible for any financial losses incurred through the use of this bot. Always test thoroughly in paper trading mode before deploying live funds.
