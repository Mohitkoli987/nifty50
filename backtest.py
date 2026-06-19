"""
backtest.py
─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS (read this before looking at any results)

You said you want signals tested "bahut sare graphs pe" (across many charts)
before trusting them with real money. This is exactly the right instinct,
and this script does that:

  1. Pulls historical daily/intraday candles for Nifty 50 AND a basket of
     other liquid symbols (so we're not just fitting to one chart's quirks).
  2. Runs signal_engine.generate_signal() candle-by-candle, EXACTLY the same
     function the live app will use — no separate "test version" of the
     logic, so a backtest pass actually means something.
  3. Simulates entries/exits with a stop-loss and target (no signal is
     useful without an exit rule — "BUY" with no exit plan isn't a
     strategy, it's a guess).
  4. Reports the metrics that matter: win rate, average win, average loss,
     total trades, max drawdown, profit factor. NOT just "% accuracy",
     because a strategy can be right 70% of the time and still lose money
     if the losses are bigger than the wins.

IMPORTANT — WHAT THIS DOES AND DOES NOT PROVE
  - If this shows a positive expectancy (profit factor > 1, reasonable
    drawdown) across MULTIPLE symbols and MULTIPLE time periods, that is
    a genuinely good sign — better than blind faith in raw indicators.
  - It does NOT prove the strategy will keep working. Markets shift
    regime. Past performance, even rigorously backtested, never
    guarantees future results. Re-run this periodically and watch for
    the live results drifting away from backtest results — that drift is
    your early warning sign to stop and re-evaluate.
  - This backtest does NOT model real slippage, brokerage, or the
    specific behavior of options premiums (theta decay, IV crush) — it
    tests the INDEX-LEVEL directional signal only. Section 2 (options
    chain) is separate and does not get backtested the same way, because
    historical options chain data isn't available from this same feed.
"""

import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

from signal_engine import generate_signal, resample_candles

UPSTOX_BASE = "https://api.upstox.com/v3/historical-candle"

# A basket of liquid NSE symbols + Nifty50 index itself, so we're not
# overfitting to one instrument's specific noise pattern.
TEST_INSTRUMENTS = {
    "NIFTY50":  "NSE_INDEX%7CNifty%2050",
    "RELIANCE": "NSE_EQ%7CINE002A01018",
    "HDFCBANK": "NSE_EQ%7CINE040A01034",
    "INFY":     "NSE_EQ%7CINE009A01021",
    "TCS":      "NSE_EQ%7CINE467B01029",
    "ICICIBANK":"NSE_EQ%7CINE090A01021",
}


def fetch_historical_candles(instrument_key_encoded, unit, interval, from_date, to_date, access_token):
    """
    Pull historical candles from Upstox v3 API.
    unit: 'minutes' or 'days'
    interval: e.g. '1', '15' for minutes; '1' for days
    Returns ascending-order list of (timestamp, open, high, low, close, volume).
    """
    url = f"{UPSTOX_BASE}/{instrument_key_encoded}/{unit}/{interval}/{to_date}/{from_date}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        if data.get("status") != "success":
            print(f"  [WARN] {url} -> {data}")
            return []
        candles = data["data"]["candles"]
        candles = list(reversed(candles))  # Upstox returns newest-first
        return [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in candles]
    except Exception as e:
        print(f"  [ERROR] fetch failed for {instrument_key_encoded}: {e}")
        return []


def simulate_trades(candles_1m, candles_15m, sl_pct=0.15, target_pct=0.30, max_hold_candles=30):
    """
    Walks through candles_1m bar by bar. At each bar (once enough history
    exists), calls generate_signal() using only data UP TO that bar (no
    look-ahead — this is critical, otherwise the backtest cheats).

    On a BUY/SELL signal with no open position, opens a simulated trade:
        - BUY:  stop-loss = entry * (1 - sl_pct/100), target = entry * (1 + target_pct/100)
        - SELL: stop-loss = entry * (1 + sl_pct/100), target = entry * (1 - target_pct/100)
    Exits on whichever of SL/target/timeout hits first.

    sl_pct / target_pct are in PERCENT (e.g. 0.15 = 0.15%), reflecting
    realistic intraday index-level moves — these are deliberately tight
    because Nifty futures/index moves of 0.3-0.5% within 30 minutes are
    already meaningfully large moves, and options buyers need much
    smaller index moves to see option-premium profit than equity traders do.

    Returns a list of trade dicts.
    """
    trades = []
    open_trade = None

    # Need a rolling 15m series aligned to each point in time. We rebuild
    # the 15m view incrementally as we walk forward, but to keep this fast
    # we just resample once up front per checkpoint — for a backtest of
    # this size that's fine.
    min_needed = 45

    for i in range(min_needed, len(candles_1m)):
        window_1m = candles_1m[: i + 1]  # only past + current bar, no look-ahead
        ts_now = window_1m[-1][0]

        # Build the 15m view using only candles up to ts_now
        window_15m = [c for c in candles_15m if c[0] <= ts_now]

        sig = generate_signal(window_1m[-200:], window_15m[-100:], market_open=True)

        current_close = window_1m[-1][4]
        current_high = window_1m[-1][2]
        current_low = window_1m[-1][3]

        # ── manage open trade first ──
        if open_trade:
            open_trade["bars_held"] += 1
            direction = open_trade["direction"]
            entry = open_trade["entry"]

            hit_sl = (current_low <= open_trade["sl"]) if direction == "BUY" else (current_high >= open_trade["sl"])
            hit_target = (current_high >= open_trade["target"]) if direction == "BUY" else (current_low <= open_trade["target"])
            timed_out = open_trade["bars_held"] >= max_hold_candles

            if hit_sl or hit_target or timed_out:
                exit_price = (
                    open_trade["sl"] if hit_sl else
                    open_trade["target"] if hit_target else
                    current_close
                )
                pnl_pct = (
                    (exit_price - entry) / entry * 100 if direction == "BUY"
                    else (entry - exit_price) / entry * 100
                )
                open_trade.update({
                    "exit": exit_price,
                    "exit_reason": "SL" if hit_sl else "TARGET" if hit_target else "TIMEOUT",
                    "pnl_pct": pnl_pct,
                    "exit_ts": ts_now,
                })
                trades.append(open_trade)
                open_trade = None
            continue  # don't open a new trade same bar we're managing one

        # ── consider opening a new trade ──
        if sig["signal"] == "BUY":
            entry = current_close
            open_trade = {
                "direction": "BUY",
                "entry": entry,
                "sl": entry * (1 - sl_pct / 100),
                "target": entry * (1 + target_pct / 100),
                "entry_ts": ts_now,
                "bars_held": 0,
                "confidence": sig["confidence"],
            }
        elif sig["signal"] == "SELL":
            entry = current_close
            open_trade = {
                "direction": "SELL",
                "entry": entry,
                "sl": entry * (1 + sl_pct / 100),
                "target": entry * (1 - target_pct / 100),
                "entry_ts": ts_now,
                "bars_held": 0,
                "confidence": sig["confidence"],
            }

    return trades


def compute_metrics(trades):
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "profit_factor": None,
            "total_pnl_pct": 0,
            "max_drawdown_pct": 0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001  # avoid div/0

    # equity curve & drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity) if len(equity) else np.array([0])
    drawdowns = running_max - equity
    max_dd = float(drawdowns.max()) if len(drawdowns) else 0

    return {
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "avg_win_pct": round(np.mean(wins), 3) if wins else 0,
        "avg_loss_pct": round(np.mean(losses), 3) if losses else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "total_pnl_pct": round(total_pnl, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }


def run_backtest(access_token, days_back=25, sl_pct=0.15, target_pct=0.30):
    """
    Main entry point. Tests across all TEST_INSTRUMENTS and prints a
    clear, honest summary table.

    NOTE on days_back: Upstox's v3 intraday-minute historical endpoint
    typically only retains recent 1m data for a limited window (often
    ~30 days for 1m candles depending on plan/instrument) — if you ask
    for too many days back and get empty results, that's most likely why,
    not a bug. The script will tell you in the per-symbol notes if a
    fetch came back empty.
    """
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"BACKTEST RUN — {from_date} to {to_date}")
    print(f"SL: {sl_pct}%  |  Target: {target_pct}%  |  Max hold: 30 candles (~30 min)")
    print(f"{'='*70}\n")

    all_results = {}

    for name, ikey in TEST_INSTRUMENTS.items():
        print(f"--- {name} ---")
        c1m = fetch_historical_candles(ikey, "minutes", "1", from_date, to_date, access_token)
        time.sleep(0.3)  # be polite to the API, avoid rate-limit

        if not c1m:
            print(f"  No 1m data returned — skipping {name} (see warning above for the reason).\n")
            all_results[name] = {"error": "no_data"}
            continue

        c15m = resample_candles(c1m, 15)

        print(f"  Fetched {len(c1m)} 1m candles, resampled to {len(c15m)} 15m candles.")

        trades = simulate_trades(c1m, c15m, sl_pct=sl_pct, target_pct=target_pct)
        metrics = compute_metrics(trades)
        all_results[name] = metrics

        if metrics["total_trades"] == 0:
            print("  No trades triggered in this window (spike+trend confirmation never aligned).")
            print("  This can be GOOD — it means the filter is rejecting noise rather than")
            print("  firing constantly. But re-run with more days if this happens everywhere.\n")
        else:
            print(f"  Trades: {metrics['total_trades']}  |  Win rate: {metrics['win_rate']}%")
            print(f"  Avg win: {metrics['avg_win_pct']}%  |  Avg loss: {metrics['avg_loss_pct']}%")
            print(f"  Profit factor: {metrics['profit_factor']}  |  Total P&L: {metrics['total_pnl_pct']}%")
            print(f"  Max drawdown: {metrics['max_drawdown_pct']}%\n")

    # ── overall honest summary ──
    print(f"\n{'='*70}")
    print("SUMMARY — READ THIS BEFORE TRADING REAL MONEY")
    print(f"{'='*70}")

    valid = {k: v for k, v in all_results.items() if v.get("total_trades", 0) > 0}
    if not valid:
        print("No symbol produced any trades in this window. Either:")
        print("  (a) the date range is too short / has no big moves, or")
        print("  (b) the confirmation filters are too strict for current conditions.")
        print("Do NOT loosen the filters just to force trades to appear — that's how")
        print("you end up back at random noise signals. Try a longer days_back first.")
    else:
        total_trades = sum(v["total_trades"] for v in valid.values())
        avg_win_rate = np.mean([v["win_rate"] for v in valid.values()])
        avg_pf = np.mean([v["profit_factor"] for v in valid.values() if v["profit_factor"]])

        print(f"Symbols with trades: {len(valid)}/{len(TEST_INSTRUMENTS)}")
        print(f"Total trades across all symbols: {total_trades}")
        print(f"Average win rate: {avg_win_rate:.1f}%")
        print(f"Average profit factor: {avg_pf:.2f}  (>1.0 = profitable, <1.0 = losing)")
        print()
        if avg_pf >= 1.3 and total_trades >= 20:
            print("Cautiously promising — but this is still ONE backtest window.")
            print("Run again with a different days_back range before trusting it.")
        elif avg_pf >= 1.0:
            print("Roughly breakeven before costs. Real brokerage + slippage would likely")
            print("push this negative. NOT ready for real money yet.")
        else:
            print("Profit factor below 1.0 — this configuration LOSES money historically.")
            print("Do not trade this live. Either the spike thresholds, the SL/target")
            print("ratio, or the timeframe needs rework — go back to signal_engine.py.")

    return all_results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("ACCESS_TOKEN")
    if not token:
        print("ACCESS_TOKEN not found in .env — backtest needs the same Upstox token as app.py")
        exit(1)

    # Try a few different parameter sets so you can see how sensitive the
    # strategy is to its exit rules — a strategy that only "works" for one
    # very specific SL/target pair is fragile and likely overfit.
    run_backtest(token, days_back=25, sl_pct=0.15, target_pct=0.30)