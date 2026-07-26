"""
Unit tests for the Order Executor.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.execution.order_executor import (
    DynamicSpreadCalculator,
    PaperTradingEngine,
    OrderParams,
    OrderSide,
    OrderStatus,
    ExecutionMode,
)


def test_spread_calculator_buy():
    calc = DynamicSpreadCalculator()
    price = calc.optimal_price(
        side=OrderSide.BUY,
        best_bid=0.40,
        best_ask=0.44,
        signal_score=0.80,
        time_to_expiry_sec=4 * 3600,
        current_spread=0.04,
    )
    assert price > 0.40, "Buy price should be above best bid"
    assert price < 0.44, "Buy price should be below best ask"
    print(f"  Spread calc (BUY) PASSED: price={price}")


def test_spread_calculator_sell():
    calc = DynamicSpreadCalculator()
    price = calc.optimal_price(
        side=OrderSide.SELL,
        best_bid=0.55,
        best_ask=0.59,
        signal_score=0.70,
        time_to_expiry_sec=12 * 3600,
        current_spread=0.04,
    )
    assert price > 0.55, "Sell price should be above best bid"
    assert price < 0.59, "Sell price should be below best ask"
    print(f"  Spread calc (SELL) PASSED: price={price}")


def test_spread_too_wide():
    calc = DynamicSpreadCalculator(max_spread_pct=0.04)
    price = calc.optimal_price(
        side=OrderSide.BUY,
        best_bid=0.30,
        best_ask=0.50,  # 20% spread!
        signal_score=0.90,
        time_to_expiry_sec=3600,
        current_spread=0.20,
    )
    assert price == -1.0, "Wide spread should return sentinel -1"
    print(f"  Wide spread rejection PASSED")


def test_paper_execution():
    paper = PaperTradingEngine(initial_balance=1000)
    params = OrderParams(
        token_id="yes_abc",
        side=OrderSide.BUY,
        price=0.52,
        size=10,
    )
    result = paper.execute(params)
    assert result.status == OrderStatus.FILLED
    assert result.cost_usd == 5.2
    assert paper.balance == 994.8
    assert paper.trades_executed == 1
    print(f"  Paper execution PASSED (balance=${paper.balance:.2f})")


def test_paper_insufficient_balance():
    paper = PaperTradingEngine(initial_balance=5)
    params = OrderParams(
        token_id="yes_xyz",
        side=OrderSide.BUY,
        price=0.60,
        size=100,  # costs $60
    )
    result = paper.execute(params)
    assert result.status == OrderStatus.CANCELLED
    print(f"  Insufficient balance test PASSED")


def test_size_calculation():
    calc = DynamicSpreadCalculator()
    shares = calc.calculate_size(5.0, 0.50)
    assert shares == 10.0, f"Expected 10 shares, got {shares}"

    shares2 = calc.calculate_size(1.0, 0.20)  # 5 shares min
    assert shares2 >= 5.0, f"Min order size violated: {shares2}"
    print(f"  Size calculation PASSED ({shares}, {shares2})")


if __name__ == "__main__":
    print("\n" + "─" * 50)
    print("  ORDER EXECUTOR TESTS")
    print("─" * 50)
    test_spread_calculator_buy()
    test_spread_calculator_sell()
    test_spread_too_wide()
    test_paper_execution()
    test_paper_insufficient_balance()
    test_size_calculation()
    print("─" * 50)
    print("  ALL TESTS PASSED\n")
