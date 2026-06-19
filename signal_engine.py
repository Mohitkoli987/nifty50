"""
signal_engine.py
─────────────────────────────────────────────────────────────────────────────
SHARED signal logic — used by BOTH backtest.py (historical testing) AND
app.py (live trading dashboard). This is intentional and important:

    If backtest.py and app.py used different code to generate signals,
    a backtest "pass" would mean nothing — you'd be testing one strategy
    and trading a different one. Every signal change must go through
    backtest.py BEFORE it ever reaches the live app.

WHAT CHANGED FROM THE OLD VERSION AND WHY
─────────────────────────────────────────────────────────────────────────────
1. OLD: Single 1-minute EMA9/EMA21 cross with no other confirmation.
   PROBLEM: 1-min candles are dominated by noise. A 0.05% wiggle crosses
   EMA9/EMA21 constantly — that's the "random buy/sell" you were seeing.
   NEW: A trade signal requires confirmation across THREE independent
   checks before it fires: (a) a real volume+price spike on 1m, (b) trend
   alignment on 15m, (c) momentum (RSI/MACD) not contradicting the move.
   If any one of these disagrees, the signal is WAIT — not a coin flip
   between BUY/SELL.

2. OLD: Score thresholds (e.g. score >= 30 = BUY) were picked by feel,
   never tested against real data.
   NEW: Thresholds are still here, but backtest.py exists specifically so
   you can SEE the historical win rate before trusting any of this with
   real money. If the backtest comes back bad, the right move is to fix
   the logic or stop — not to keep tweaking numbers until backtest looks
   good (that's overfitting and it WILL fail live).

3. NEW: "Spike" detection. A spike = price moving >= N standard deviations
   of its recent range, on volume that is also above its recent average.
   This filters out the slow noisy drift you were complaining about and
   only signals on moves that are actually large enough to matter.

4. STILL TRUE: No signal here is a guarantee. This is pattern detection on
   historical statistics, not a prediction of the future. Markets change
   regime; a setup that worked for the last 6 months may stop working.
   That is why position sizing / stop-loss discipline matters more than
   the signal itself.
"""

import math
import numpy as np


# ─── BASIC INDICATORS (unchanged math, just cleaned up) ──────────────────

def ema(values, period):
    """Exponential moving average. Returns list aligned to the END of `values`
    (i.e. ema(values, p)[-1] corresponds to values[-1])."""
    if len(values) < period:
        return [None] * len(values)
    out = [None] * (period - 1)
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    out.append(e)
    for v in values[period:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(closes, period=14):
    if len(closes) < period + 1:
        return [None] * len(closes)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    out = [None] * period
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    out.append(100 - 100 / (1 + rs))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        out.append(100 - 100 / (1 + rs))
    return out


def macd(closes, fast=12, slow=26, signal=9):
    e_fast = ema(closes, fast)
    e_slow = ema(closes, slow)
    macd_line = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(e_fast, e_slow)
    ]
    valid = [v for v in macd_line if v is not None]
    sig_valid = ema(valid, signal)
    # re-pad to align with macd_line length
    pad = len(macd_line) - len(sig_valid)
    signal_line = [None] * pad + sig_valid
    hist = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, hist


def atr(highs, lows, closes, period=14):
    trs = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    out = [None] * (period - 1)
    a = sum(trs[:period]) / period
    out.append(a)
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
        out.append(a)
    return out


def supertrend(highs, lows, closes, period=10, multiplier=3):
    """Returns direction list: 1 = bullish, -1 = bearish, None where undefined."""
    n = len(closes)
    atr_vals = atr(highs, lows, closes, period)
    direction = [None] * n
    final_ub = [None] * n
    final_lb = [None] * n

    for i in range(n):
        if atr_vals[i] is None:
            continue
        hl2 = (highs[i] + lows[i]) / 2
        ub = hl2 + multiplier * atr_vals[i]
        lb = hl2 - multiplier * atr_vals[i]

        prev_idx = i - 1
        if prev_idx < 0 or final_ub[prev_idx] is None:
            final_ub[i] = ub
            final_lb[i] = lb
            direction[i] = 1 if closes[i] > ub else -1
            continue

        final_ub[i] = ub if (ub < final_ub[prev_idx] or closes[prev_idx] > final_ub[prev_idx]) else final_ub[prev_idx]
        final_lb[i] = lb if (lb > final_lb[prev_idx] or closes[prev_idx] < final_lb[prev_idx]) else final_lb[prev_idx]

        prev_dir = direction[prev_idx]
        if prev_dir == 1:
            direction[i] = -1 if closes[i] < final_lb[i] else 1
        else:
            direction[i] = 1 if closes[i] > final_ub[i] else -1

    return direction


def vwap(highs, lows, closes, volumes):
    cum_pv, cum_v = 0.0, 0.0
    out = []
    for h, l, c, v in zip(highs, lows, closes, volumes):
        typ = (h + l + c) / 3
        cum_pv += typ * v
        cum_v += v
        out.append(cum_pv / cum_v if cum_v > 0 else typ)
    return out


# ─── SPIKE DETECTION ──────────────────────────────────────────────────────
# This is the direct answer to "1 minute ka bada spike chahiye" —
# we don't react to every tick, only to moves that are statistically
# unusual in BOTH price range and volume.

def detect_spikes(highs, lows, closes, volumes, lookback=20, price_z=1.5, vol_mult=1.5):
    """
    For each candle i (i >= lookback), compute:
      - candle range = high[i] - low[i]
      - rolling mean/std of range over the previous `lookback` candles
      - rolling mean of volume over the previous `lookback` candles
    A candle is flagged as a spike if:
      - its range is >= price_z standard deviations above the rolling mean range
      - AND its volume is >= vol_mult times the rolling mean volume
      - AND it closes in the direction of the move (close near the high for
        an up-spike, near the low for a down-spike) — filters out big-range
        indecisive candles (long wicks both sides) from real breakout moves.

    Returns a list the same length as closes: 1 = bullish spike,
    -1 = bearish spike, 0 = no spike.
    """
    n = len(closes)
    spikes = [0] * n
    ranges = [h - l for h, l in zip(highs, lows)]

    for i in range(lookback, n):
        window_ranges = ranges[i - lookback:i]
        window_vols = volumes[i - lookback:i]

        mean_r = np.mean(window_ranges)
        std_r = np.std(window_ranges)
        mean_v = np.mean(window_vols) if np.mean(window_vols) > 0 else 1

        if std_r == 0:
            continue

        this_range = ranges[i]
        z = (this_range - mean_r) / std_r

        if z >= price_z and volumes[i] >= vol_mult * mean_v:
            candle_pos = (closes[i] - lows[i]) / this_range if this_range > 0 else 0.5
            if closes[i] >= highs[i] * 0.999 or candle_pos >= 0.65:
                spikes[i] = 1   # bullish spike — closed strong near the high
            elif closes[i] <= lows[i] * 1.001 or candle_pos <= 0.35:
                spikes[i] = -1  # bearish spike — closed weak near the low

    return spikes


# ─── MULTI-TIMEFRAME CONFIRMATION ────────────────────────────────────────

def resample_candles(candles_1m, minutes=15):
    """
    candles_1m: list of dicts/tuples (timestamp, open, high, low, close, volume)
    in ASCENDING time order. Aggregates into `minutes`-minute candles.
    Expects candle[0] to be a pandas-parseable timestamp string or datetime.
    """
    import pandas as pd
    if not candles_1m:
        return []
    df = pd.DataFrame(candles_1m, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    agg = df.resample(f"{minutes}min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    agg = agg.reset_index()
    return list(agg.itertuples(index=False, name=None))


# ─── MAIN SIGNAL FUNCTION ─────────────────────────────────────────────────

def generate_signal(candles_1m, candles_15m, market_open=True, min_candles=40):
    """
    candles_1m / candles_15m: list of tuples/lists
        (timestamp, open, high, low, close, volume) in ASCENDING order (oldest first).

    Returns dict:
        {
            "signal": "BUY" | "SELL" | "WAIT" | "CLOSED",
            "confidence": 0-100,
            "reasons": [...],
            "indicators": {...},
            "spike": 1 | -1 | 0,
        }

    LOGIC — all three must agree for a non-WAIT signal:
      1. SPIKE on 1m: a statistically large price+volume move just happened.
      2. TREND on 15m: EMA + Supertrend on the higher timeframe agree with
         the spike direction. This is the main fix for "kabhi buy kabhi
         sell" — a 1-minute spike against the 15-minute trend is usually
         a fakeout / liquidity grab, not a real move. We WAIT on those.
      3. MOMENTUM not contradicting: RSI not already extreme in the
         opposite direction (e.g. don't BUY into RSI > 80).
    """
    result = {
        "signal": "WAIT",
        "confidence": 0,
        "reasons": [],
        "indicators": {},
        "spike": 0,
    }

    if not market_open:
        result["signal"] = "CLOSED"
        result["reasons"] = ["Market closed (9:15-15:30 IST, Mon-Fri)"]
        return result

    if not candles_1m or len(candles_1m) < min_candles:
        result["reasons"] = [f"Not enough 1m data (have {len(candles_1m) if candles_1m else 0}, need {min_candles})"]
        return result

    closes_1m = [c[4] for c in candles_1m]
    highs_1m = [c[2] for c in candles_1m]
    lows_1m = [c[3] for c in candles_1m]
    vols_1m = [c[5] for c in candles_1m]

    spikes = detect_spikes(highs_1m, lows_1m, closes_1m, vols_1m)
    current_spike = spikes[-1]

    rsi_1m = rsi(closes_1m, 14)
    macd_line, macd_sig, macd_hist = macd(closes_1m)

    result["indicators"] = {
        "price": round(closes_1m[-1], 2),
        "rsi_1m": round(rsi_1m[-1], 2) if rsi_1m[-1] is not None else None,
        "macd_hist_1m": round(macd_hist[-1], 2) if macd_hist[-1] is not None else None,
    }
    result["spike"] = current_spike

    if current_spike == 0:
        result["reasons"] = ["No significant 1m spike detected — waiting for a real move, not noise"]
        return result

    # ── 15m trend confirmation ──
    if not candles_15m or len(candles_15m) < 25:
        result["reasons"] = ["1m spike detected but not enough 15m data to confirm trend — staying WAIT for safety"]
        return result

    closes_15m = [c[4] for c in candles_15m]
    highs_15m = [c[2] for c in candles_15m]
    lows_15m = [c[3] for c in candles_15m]

    ema9_15 = ema(closes_15m, 9)
    ema21_15 = ema(closes_15m, 21)
    st_dir_15 = supertrend(highs_15m, lows_15m, closes_15m, 10, 3)

    trend_15m_bullish = (
        ema9_15[-1] is not None and ema21_15[-1] is not None and
        ema9_15[-1] > ema21_15[-1] and st_dir_15[-1] == 1
    )
    trend_15m_bearish = (
        ema9_15[-1] is not None and ema21_15[-1] is not None and
        ema9_15[-1] < ema21_15[-1] and st_dir_15[-1] == -1
    )

    result["indicators"]["trend_15m"] = (
        "BULLISH" if trend_15m_bullish else "BEARISH" if trend_15m_bearish else "MIXED"
    )

    reasons = []
    confidence = 40  # base confidence once a real spike fires

    if current_spike == 1:
        reasons.append("Bullish 1m spike (price+volume above normal range, closed strong)")
        if not trend_15m_bullish:
            result["reasons"] = reasons + ["15m trend does NOT confirm bullish move — likely a fakeout, WAIT"]
            return result
        reasons.append("15m trend confirms bullish (EMA9>EMA21 + Supertrend bullish)")
        confidence += 25

        if rsi_1m[-1] is not None and rsi_1m[-1] >= 80:
            result["reasons"] = reasons + [f"RSI already extreme ({rsi_1m[-1]:.1f}) — too late to chase, WAIT"]
            return result
        if rsi_1m[-1] is not None and 45 <= rsi_1m[-1] <= 75:
            reasons.append(f"RSI healthy ({rsi_1m[-1]:.1f}), room to run")
            confidence += 15

        if macd_hist[-1] is not None and macd_hist[-1] > 0:
            reasons.append("MACD histogram positive — momentum agrees")
            confidence += 15

        result["signal"] = "BUY"

    elif current_spike == -1:
        reasons.append("Bearish 1m spike (price+volume above normal range, closed weak)")
        if not trend_15m_bearish:
            result["reasons"] = reasons + ["15m trend does NOT confirm bearish move — likely a fakeout, WAIT"]
            return result
        reasons.append("15m trend confirms bearish (EMA9<EMA21 + Supertrend bearish)")
        confidence += 25

        if rsi_1m[-1] is not None and rsi_1m[-1] <= 20:
            result["reasons"] = reasons + [f"RSI already extreme ({rsi_1m[-1]:.1f}) — too late to chase, WAIT"]
            return result
        if rsi_1m[-1] is not None and 25 <= rsi_1m[-1] <= 55:
            reasons.append(f"RSI healthy ({rsi_1m[-1]:.1f}), room to fall")
            confidence += 15

        if macd_hist[-1] is not None and macd_hist[-1] < 0:
            reasons.append("MACD histogram negative — momentum agrees")
            confidence += 15

        result["signal"] = "SELL"

    result["confidence"] = min(confidence, 95)
    result["reasons"] = reasons
    return result