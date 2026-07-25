#!/usr/bin/env python3
"""
Polymarket Alpha Bot – Entry Point
===================================
AI vs Humans + Cross-Platform Fair Value + Whale Shadowing

Usage:
  Paper trading (default):
    python run.py

  Live trading (tiny size):
    BOT_ENV=live python run.py

  Custom scan parameters:
    python run.py --markets 50 --iterations 5

  Single market evaluation:
    python run.py --market <condition_id>

  Quick demo (no API calls):
    python run.py --demo
"""
import argparse
import logging
import sys

# Ensure project root is on path
sys.path.insert(0, ".")

from config.settings import ENV, LOG_LEVEL
from src.core.orchestrator import Orchestrator


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Polymarket Alpha Bot")
    parser.add_argument("--markets", type=int, default=20, help="Markets per scan")
    parser.add_argument("--iterations", type=int, default=10, help="Number of scan iterations")
    parser.add_argument("--market", type=str, default=None, help="Single market condition_id")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with mock data")
    parser.add_argument("--env", type=str, default=ENV, help="Override mode: paper|live")
    return parser.parse_args()


def run_demo():
    """Run a demo with synthetic data – no API calls needed."""
    from src.signals.signal_engine import SignalEngine, SignalDirection

    engine = SignalEngine()

    # Simulate a market with strong divergence
    signal = engine.compute(
        condition_id="0xabc123demo",
        question="Will BTC reach $200k by Dec 31, 2026?",
        market_prob=0.35,          # Human market says 35%
        ai_divergence=0.22,         # AI consensus says ~57%
        ai_confidence=0.78,
        fair_value=0.52,            # Crypto fair value model says 52%
        whale_score=0.65,
        whale_count=4,
        whale_direction="BUY",
        time_to_expiry_sec=7 * 86400,  # 7 days
    )

    print("\n" + "=" * 60)
    print("  DEMO SIGNAL OUTPUT")
    print("=" * 60)
    print(signal.summary())
    print()

    # Simulate a weak signal
    signal2 = engine.compute(
        condition_id="0xdef456demo",
        question="Will it rain tomorrow in NYC?",
        market_prob=0.51,
        ai_divergence=0.02,
        ai_confidence=0.30,
        fair_value=None,
        whale_score=0.05,
        whale_count=0,
        whale_direction="NEUTRAL",
        time_to_expiry_sec=86400 * 30,
    )
    print(signal2.summary())
    print()

    # Demo order executor
    from src.execution.order_executor import PaperTradingEngine, DynamicSpreadCalculator, OrderSide

    paper = PaperTradingEngine(initial_balance=10_000)
    spread_calc = DynamicSpreadCalculator()

    price = spread_calc.optimal_price(
        side=OrderSide.BUY,
        best_bid=0.34,
        best_ask=0.37,
        signal_score=0.85,
        time_to_expiry_sec=7 * 86400,
        current_spread=0.03,
    )
    size = spread_calc.calculate_size(5.0, price)

    print(f"  Optimal bid price: {price}")
    print(f"  Order size:        {size} shares")
    print(f"  Estimated cost:    ${price * size:.2f}")

    from src.execution.order_executor import OrderParams, ExecutionMode

    params = OrderParams(token_id="yes_token", side=OrderSide.BUY, price=price, size=size)
    result = paper.execute(params)
    print(f"\n  Order result: {result.status.value} | Cost: ${result.cost_usd:.2f}")
    print(paper.summary())


def main():
    setup_logging()
    args = parse_args()

    if args.env:
        import os
        os.environ["BOT_ENV"] = args.env

    if args.demo:
        print("\n" + "━" * 60)
        print("  POLYMARKET ALPHA BOT — DEMO MODE")
        print("━" * 60)
        run_demo()
        return

    orchestrator = Orchestrator()

    if args.market:
        # Single market evaluation
        print(f"Evaluating market: {args.market}")
        market_data = orchestrator.pm.market_details(args.market)
        if market_data:
            traded = orchestrator.evaluate_market(market_data)
            if traded:
                print("TRADE EXECUTED!")
            else:
                print("No trade triggered.")
        else:
            print(f"Market {args.market} not found or inactive.")
    else:
        # Full scan loop
        orchestrator.run(num_markets=args.markets, iterations=args.iterations)


if __name__ == "__main__":
    main()
