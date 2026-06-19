from flask import Flask, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import json
import math
import socket
import logging
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ─── LOGGING SETUP ────────────────────────────────────────────────────────
# This is the main thing that was missing before: every fetch function was
# only doing `print(f"... error: {e}")`, which (a) only fires on a raw
# exception — NOT on a bad status code or an API error payload — and
# (b) doesn't tell you WHAT Upstox actually said. Now everything goes
# through `logger`, prints to console AND writes to app.log, and we log
# status codes + full response bodies even when there's no exception.

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),                       # prints to terminal
        logging.FileHandler(LOG_FILE, encoding="utf-8") # also saves to app.log
    ]
)
logger = logging.getLogger("nifty_app")

load_dotenv()

app = Flask(__name__)
CORS(app)

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
INSTRUMENT_KEY_ENCODED = "NSE_INDEX%7CNifty%2050"

if not ACCESS_TOKEN:
    logger.error(
        "ACCESS_TOKEN is missing/empty! Check that a .env file exists next to "
        "app.py and contains ACCESS_TOKEN=<your_token>. Every Upstox call will "
        "fail with 401 Unauthorized until this is fixed."
    )
else:
    # don't log the full token (sensitive), just confirm it's loaded + length
    logger.info(f"ACCESS_TOKEN loaded (length={len(ACCESS_TOKEN)} chars).")

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

# ─── TECHNICAL INDICATORS ────────────────────────────────────────────────────

def compute_ema(closes, period):
    if len(closes) < period:
        return []
    emas = []
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    emas.append(ema)
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
        emas.append(ema)
    return emas

def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi_vals = []
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi_vals.append(100 - (100 / (1 + rs)))
    return rsi_vals

def compute_macd(closes):
    ema12 = compute_ema(closes, 12)
    ema26 = compute_ema(closes, 26)
    if not ema12 or not ema26:
        return [], [], []
    # align lengths
    diff = len(ema12) - len(ema26)
    ema12 = ema12[diff:]
    macd_line = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
    signal_line = compute_ema(macd_line, 9)
    diff2 = len(macd_line) - len(signal_line)
    macd_trimmed = macd_line[diff2:]
    histogram = [m - s for m, s in zip(macd_trimmed, signal_line)]
    return macd_trimmed, signal_line, histogram

def compute_supertrend(candles, period=7, multiplier=3):
    if len(candles) < period:
        return []
    highs = [c[2] for c in candles]
    lows  = [c[3] for c in candles]
    closes= [c[4] for c in candles]

    # ATR
    tr_list = []
    for i in range(1, len(candles)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        tr_list.append(tr)

    atr = []
    atr.append(sum(tr_list[:period]) / period)
    for i in range(period, len(tr_list)):
        atr.append((atr[-1] * (period - 1) + tr_list[i]) / period)

    start = period  # candle index where atr starts

    upperband = []
    lowerband = []
    supertrend = []
    direction = []

    for i in range(len(atr)):
        ci = start + i
        hl2 = (highs[ci] + lows[ci]) / 2
        ub = hl2 + multiplier * atr[i]
        lb = hl2 - multiplier * atr[i]

        if i == 0:
            upperband.append(ub)
            lowerband.append(lb)
            supertrend.append(ub)
            direction.append(-1)
        else:
            # adjust bands
            final_ub = ub if ub < upperband[-1] or closes[ci-1] > upperband[-1] else upperband[-1]
            final_lb = lb if lb > lowerband[-1] or closes[ci-1] < lowerband[-1] else lowerband[-1]
            upperband.append(final_ub)
            lowerband.append(final_lb)

            if direction[-1] == -1:
                st = final_lb if closes[ci] > supertrend[-1] else final_ub
                d = 1 if closes[ci] > supertrend[-1] else -1
            else:
                st = final_ub if closes[ci] < supertrend[-1] else final_lb
                d = -1 if closes[ci] < supertrend[-1] else 1

            supertrend.append(st)
            direction.append(d)

    return direction  # 1 = bullish, -1 = bearish

def compute_vwap(candles):
    cum_pv = 0
    cum_v = 0
    vwaps = []
    for c in candles:
        typ = (c[2] + c[3] + c[4]) / 3
        vol = c[5]
        cum_pv += typ * vol
        cum_v += vol
        vwaps.append(cum_pv / cum_v if cum_v > 0 else typ)
    return vwaps

def compute_bollinger(closes, period=20, std_dev=2):
    if len(closes) < period:
        return [], [], []
    upper, mid, lower = [], [], []
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        m = sum(window) / period
        sd = math.sqrt(sum((x - m) ** 2 for x in window) / period)
        upper.append(m + std_dev * sd)
        mid.append(m)
        lower.append(m - std_dev * sd)
    return upper, mid, lower

def compute_stochastic(candles, k_period=14, d_period=3):
    if len(candles) < k_period:
        return [], []
    k_vals = []
    for i in range(k_period - 1, len(candles)):
        window = candles[i - k_period + 1 : i + 1]
        h = max(c[2] for c in window)
        l = min(c[3] for c in window)
        close = candles[i][4]
        k = ((close - l) / (h - l) * 100) if (h - l) != 0 else 50
        k_vals.append(k)
    d_vals = []
    for i in range(d_period - 1, len(k_vals)):
        d_vals.append(sum(k_vals[i - d_period + 1 : i + 1]) / d_period)
    return k_vals, d_vals

# ─── SIGNAL ENGINE ───────────────────────────────────────────────────────────

def generate_signals(candles_1m, candles_15m, market_open=True):
    signals = {
        "scalping": {"signal": "WAIT", "confidence": 0, "reason": []},
        "swing":    {"signal": "WAIT", "confidence": 0, "reason": []},
        "hold":     {"signal": "WAIT", "confidence": 0, "reason": []},
        "trend":    "SIDEWAYS",
        "indicators": {}
    }

    closed_reason = ["Market is closed (9:15 AM – 3:30 PM IST, Mon–Fri)"]

    if not candles_1m or len(candles_1m) < 30:
        logger.warning(
            f"generate_signals: not enough 1m candle data "
            f"(got {len(candles_1m) if candles_1m else 0}, need >= 30). "
            f"Signals will stay as defaults/CLOSED."
        )
        # Not enough data to compute anything meaningful.
        if not market_open:
            signals["scalping"] = {"signal": "CLOSED", "confidence": 0, "reason": closed_reason}
            signals["swing"]    = {"signal": "CLOSED", "confidence": 0, "reason": closed_reason}
            signals["hold"]     = {"signal": "CLOSED", "confidence": 0, "reason": closed_reason}
            signals["trend"]    = "CLOSED"
        return signals

    closes_1m = [c[4] for c in candles_1m]
    closes_15m = [c[4] for c in candles_15m] if candles_15m else closes_1m

    # ── Indicators on 1m ──
    ema9  = compute_ema(closes_1m, 9)
    ema21 = compute_ema(closes_1m, 21)
    ema50 = compute_ema(closes_1m, 50)
    rsi   = compute_rsi(closes_1m, 14)
    macd_line, signal_line, histogram = compute_macd(closes_1m)
    vwap  = compute_vwap(candles_1m)
    bb_u, bb_m, bb_l = compute_bollinger(closes_1m, 20)
    st_dir = compute_supertrend(candles_1m, 7, 3)
    k_vals, d_vals = compute_stochastic(candles_1m)

    # ── Indicators on 15m ──
    ema9_15  = compute_ema(closes_15m, 9)
    ema21_15 = compute_ema(closes_15m, 21)
    rsi_15   = compute_rsi(closes_15m, 14)
    st_dir_15= compute_supertrend(candles_15m, 7, 3) if len(candles_15m) >= 10 else st_dir

    price = closes_1m[-1]

    # Store latest indicator values for dashboard
    signals["indicators"] = {
        "price":      round(price, 2),
        "ema9":       round(ema9[-1], 2)  if ema9  else None,
        "ema21":      round(ema21[-1], 2) if ema21 else None,
        "ema50":      round(ema50[-1], 2) if ema50 else None,
        "rsi":        round(rsi[-1], 2)   if rsi   else None,
        "macd":       round(macd_line[-1], 2) if macd_line else None,
        "macd_signal":round(signal_line[-1], 2) if signal_line else None,
        "histogram":  round(histogram[-1], 2) if histogram else None,
        "vwap":       round(vwap[-1], 2)  if vwap  else None,
        "bb_upper":   round(bb_u[-1], 2)  if bb_u  else None,
        "bb_lower":   round(bb_l[-1], 2)  if bb_l  else None,
        "stoch_k":    round(k_vals[-1], 2) if k_vals else None,
        "stoch_d":    round(d_vals[-1], 2) if d_vals else None,
        "supertrend": "BULL" if (st_dir and st_dir[-1] == 1) else "BEAR",
        "supertrend_15m": "BULL" if (st_dir_15 and st_dir_15[-1] == 1) else "BEAR",
    }

    # ── TREND determination ──
    bullish_count = 0
    bearish_count = 0

    if ema9 and ema21 and ema9[-1] > ema21[-1]: bullish_count += 1
    else: bearish_count += 1

    if ema21 and ema50 and ema21[-1] > ema50[-1]: bullish_count += 1
    else: bearish_count += 1

    if st_dir and st_dir[-1] == 1: bullish_count += 2
    else: bearish_count += 2

    if st_dir_15 and st_dir_15[-1] == 1: bullish_count += 2
    else: bearish_count += 2

    if rsi and rsi[-1] > 55: bullish_count += 1
    elif rsi and rsi[-1] < 45: bearish_count += 1

    if bullish_count >= 5:
        signals["trend"] = "BULLISH"
    elif bearish_count >= 5:
        signals["trend"] = "BEARISH"
    else:
        signals["trend"] = "SIDEWAYS"

    # ─── SCALPING SIGNAL (1-5 min) ───
    scalp_score = 0
    scalp_reasons = []

    if ema9 and ema21:
        if ema9[-1] > ema21[-1] and (len(ema9) < 2 or len(ema21) < 2 or ema9[-2] <= ema21[-2]):
            scalp_score += 30
            scalp_reasons.append("EMA9 crossed above EMA21")
        elif ema9[-1] < ema21[-1] and (len(ema9) < 2 or len(ema21) < 2 or ema9[-2] >= ema21[-2]):
            scalp_score -= 30
            scalp_reasons.append("EMA9 crossed below EMA21")
        elif ema9[-1] > ema21[-1]:
            scalp_score += 10

    if rsi:
        r = rsi[-1]
        if 40 < r < 60:
            scalp_score += 5
            scalp_reasons.append(f"RSI neutral {r:.1f}")
        elif r > 70:
            scalp_score -= 20
            scalp_reasons.append(f"RSI overbought {r:.1f}")
        elif r < 30:
            scalp_score += 20
            scalp_reasons.append(f"RSI oversold {r:.1f}")

    if macd_line and signal_line and histogram:
        if histogram[-1] > 0 and (len(histogram) < 2 or histogram[-2] <= 0):
            scalp_score += 25
            scalp_reasons.append("MACD histogram turned positive")
        elif histogram[-1] < 0 and (len(histogram) < 2 or histogram[-2] >= 0):
            scalp_score -= 25
            scalp_reasons.append("MACD histogram turned negative")

    if vwap and price > vwap[-1]:
        scalp_score += 10
        scalp_reasons.append("Price above VWAP")
    elif vwap:
        scalp_score -= 10

    if bb_u and bb_l and bb_m:
        if price <= bb_l[-1]:
            scalp_score += 15
            scalp_reasons.append("Price at BB lower band — bounce possible")
        elif price >= bb_u[-1]:
            scalp_score -= 15
            scalp_reasons.append("Price at BB upper band — reversal possible")

    if st_dir and st_dir[-1] == 1:
        scalp_score += 15
        scalp_reasons.append("Supertrend Bullish (1m)")
    elif st_dir:
        scalp_score -= 15

    confidence = min(abs(scalp_score), 95)
    if scalp_score >= 30:
        signals["scalping"] = {"signal": "BUY", "confidence": confidence, "reason": scalp_reasons}
    elif scalp_score <= -30:
        signals["scalping"] = {"signal": "SELL", "confidence": confidence, "reason": scalp_reasons}
    else:
        signals["scalping"] = {"signal": "WAIT", "confidence": confidence, "reason": scalp_reasons}

    # ─── SWING SIGNAL (15-30 min hold) ───
    swing_score = 0
    swing_reasons = []

    if ema9_15 and ema21_15:
        if ema9_15[-1] > ema21_15[-1]:
            swing_score += 25
            swing_reasons.append("15m EMA9 > EMA21 — uptrend")
        else:
            swing_score -= 25
            swing_reasons.append("15m EMA9 < EMA21 — downtrend")

    if st_dir_15 and st_dir_15[-1] == 1:
        swing_score += 30
        swing_reasons.append("15m Supertrend Bullish")
    elif st_dir_15:
        swing_score -= 30
        swing_reasons.append("15m Supertrend Bearish")

    if rsi_15:
        r15 = rsi_15[-1]
        if r15 > 60:
            swing_score += 15
            swing_reasons.append(f"15m RSI strong {r15:.1f}")
        elif r15 < 40:
            swing_score -= 15
            swing_reasons.append(f"15m RSI weak {r15:.1f}")

    if vwap and price > vwap[-1]:
        swing_score += 15
        swing_reasons.append("Above VWAP — buyers in control")
    elif vwap:
        swing_score -= 15

    conf_sw = min(abs(swing_score), 95)
    if swing_score >= 40:
        signals["swing"] = {"signal": "BUY", "confidence": conf_sw, "reason": swing_reasons}
    elif swing_score <= -40:
        signals["swing"] = {"signal": "SELL", "confidence": conf_sw, "reason": swing_reasons}
    else:
        signals["swing"] = {"signal": "WAIT", "confidence": conf_sw, "reason": swing_reasons}

    # ─── HOLD / POSITIONAL SIGNAL ───
    hold_score = 0
    hold_reasons = []

    if ema50 and price > ema50[-1]:
        hold_score += 30
        hold_reasons.append("Price above 50 EMA")
    elif ema50:
        hold_score -= 30

    if bullish_count > bearish_count:
        hold_score += 20
        hold_reasons.append(f"Trend consensus bullish ({bullish_count} vs {bearish_count})")
    elif bearish_count > bullish_count:
        hold_score -= 20
        hold_reasons.append(f"Trend consensus bearish ({bearish_count} vs {bullish_count})")

    if rsi and rsi[-1] > 50:
        hold_score += 15
        hold_reasons.append("RSI above 50 — momentum positive")
    elif rsi:
        hold_score -= 15

    conf_h = min(abs(hold_score), 95)
    if hold_score >= 35:
        signals["hold"] = {"signal": "BUY", "confidence": conf_h, "reason": hold_reasons}
    elif hold_score <= -35:
        signals["hold"] = {"signal": "SELL", "confidence": conf_h, "reason": hold_reasons}
    else:
        signals["hold"] = {"signal": "WAIT", "confidence": conf_h, "reason": hold_reasons}

    # ─── MARKET CLOSED OVERRIDE ───
    # Indicators above stay populated for reference (last session's values), but we
    # NEVER hand out a fresh BUY/SELL/WAIT call while the market isn't actually trading —
    # that's exactly the "fake signal" problem. Force a clear CLOSED state instead.
    if not market_open:
        signals["scalping"] = {"signal": "CLOSED", "confidence": 0, "reason": closed_reason}
        signals["swing"]    = {"signal": "CLOSED", "confidence": 0, "reason": closed_reason}
        signals["hold"]     = {"signal": "CLOSED", "confidence": 0, "reason": closed_reason}
        signals["trend"]    = "CLOSED"

    return signals


# ─── DATA FETCHING ───────────────────────────────────────────────────────────
# Every function below now:
#   1. Logs the exact URL it's hitting
#   2. Logs the HTTP status code it got back
#   3. Logs the raw response body if status != success/200 (so you can see
#      Upstox's actual error message, e.g. "Invalid token", "Instrument key
#      not found", "Too many requests", etc.)
#   4. Logs full traceback on exceptions instead of just the exception text

def fetch_intraday(unit="minutes", interval="1"):
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{INSTRUMENT_KEY_ENCODED}/{unit}/{interval}"
    logger.debug(f"fetch_intraday: GET {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        logger.debug(f"fetch_intraday: status_code={r.status_code}")
        try:
            data = r.json()
        except ValueError:
            logger.error(f"fetch_intraday: response was not valid JSON. Raw text: {r.text[:500]}")
            return []

        if data.get("status") == "success":
            candles = data["data"]["candles"]
            logger.info(f"fetch_intraday: got {len(candles)} candles for {unit}/{interval}")
            return candles
        else:
            logger.error(
                f"fetch_intraday: Upstox returned non-success status. "
                f"HTTP={r.status_code}, body={json.dumps(data)[:1000]}"
            )
    except requests.exceptions.Timeout:
        logger.error(f"fetch_intraday: request timed out after 10s ({url})")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"fetch_intraday: connection error — check internet/firewall. {e}")
    except Exception as e:
        logger.error(f"fetch_intraday: unexpected error: {e}\n{traceback.format_exc()}")
    return []

def fetch_historical(unit="days", interval="1", days_back=90):
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = f"https://api.upstox.com/v3/historical-candle/{INSTRUMENT_KEY_ENCODED}/{unit}/{interval}/{to_date}/{from_date}"
    logger.debug(f"fetch_historical: GET {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        logger.debug(f"fetch_historical: status_code={r.status_code}")
        try:
            data = r.json()
        except ValueError:
            logger.error(f"fetch_historical: response was not valid JSON. Raw text: {r.text[:500]}")
            return []

        if data.get("status") == "success":
            candles = data["data"]["candles"]
            logger.info(f"fetch_historical: got {len(candles)} candles")
            return candles
        else:
            logger.error(
                f"fetch_historical: Upstox returned non-success status. "
                f"HTTP={r.status_code}, body={json.dumps(data)[:1000]}"
            )
    except requests.exceptions.Timeout:
        logger.error(f"fetch_historical: request timed out after 10s ({url})")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"fetch_historical: connection error — check internet/firewall. {e}")
    except Exception as e:
        logger.error(f"fetch_historical: unexpected error: {e}\n{traceback.format_exc()}")
    return []

def fetch_ltp():
    """
    Fetch the real, live Last Traded Price for Nifty 50 directly from Upstox's
    LTP quote endpoint. This works whether the market is open or closed (it
    returns the last traded price either way), so the dashboard always shows
    a real price instead of '--' when intraday candles aren't available yet.
    """
    url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={INSTRUMENT_KEY_ENCODED}"
    logger.debug(f"fetch_ltp: GET {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        logger.debug(f"fetch_ltp: status_code={r.status_code}")
        try:
            data = r.json()
        except ValueError:
            logger.error(f"fetch_ltp: response was not valid JSON. Raw text: {r.text[:500]}")
            return None

        if data.get("status") == "success":
            quote_data = data.get("data", {})
            if not quote_data:
                logger.warning(f"fetch_ltp: status=success but 'data' object is empty. Full body: {json.dumps(data)[:1000]}")
            for key, val in quote_data.items():
                if isinstance(val, dict) and "last_price" in val and val["last_price"] is not None:
                    logger.info(f"fetch_ltp: got last_price={val['last_price']} for key={key}")
                    return val["last_price"]
            logger.warning(f"fetch_ltp: no 'last_price' field found in any quote entry. body={json.dumps(data)[:1000]}")
        else:
            logger.error(
                f"fetch_ltp: Upstox returned non-success status. "
                f"HTTP={r.status_code}, body={json.dumps(data)[:1000]}"
            )
    except requests.exceptions.Timeout:
        logger.error(f"fetch_ltp: request timed out after 10s ({url})")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"fetch_ltp: connection error — check internet/firewall. {e}")
    except Exception as e:
        logger.error(f"fetch_ltp: unexpected error: {e}\n{traceback.format_exc()}")
    return None

def fetch_wallet_balance():
    """
    Fetch the user's available trading margin/funds (wallet balance) from
    Upstox's funds-and-margin endpoint.
    """
    url = "https://api.upstox.com/v2/user/get-funds-and-margin"
    logger.debug(f"fetch_wallet_balance: GET {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        logger.debug(f"fetch_wallet_balance: status_code={r.status_code}")
        try:
            data = r.json()
        except ValueError:
            logger.error(f"fetch_wallet_balance: response was not valid JSON. Raw text: {r.text[:500]}")
            return {"available_margin": None, "used_margin": None}

        if data.get("status") == "success":
            equity = data.get("data", {}).get("equity", {})
            if not equity:
                logger.warning(
                    f"fetch_wallet_balance: status=success but no 'equity' key in data. "
                    f"This usually means the account has no equity segment activated, "
                    f"or the token's scope doesn't include funds. Full body: {json.dumps(data)[:1000]}"
                )
            result = {
                "available_margin": equity.get("available_margin"),
                "used_margin":      equity.get("used_margin"),
            }
            logger.info(f"fetch_wallet_balance: {result}")
            return result
        else:
            logger.error(
                f"fetch_wallet_balance: Upstox returned non-success status (likely 401 "
                f"Unauthorized if token expired, or 403 if scope/permission issue). "
                f"HTTP={r.status_code}, body={json.dumps(data)[:1000]}"
            )
    except requests.exceptions.Timeout:
        logger.error(f"fetch_wallet_balance: request timed out after 10s ({url})")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"fetch_wallet_balance: connection error — check internet/firewall. {e}")
    except Exception as e:
        logger.error(f"fetch_wallet_balance: unexpected error: {e}\n{traceback.format_exc()}")
    return {"available_margin": None, "used_margin": None}

def get_local_ip():
    """
    Get the local network IP address of the machine running this Flask app
    (e.g. 192.168.x.x). Useful so you can open the dashboard from your phone
    or another device on the same Wi-Fi/LAN using http://<this_ip>:5000
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This doesn't actually send any packets — connect() on a UDP socket
        # just picks which local interface/IP would be used to reach 8.8.8.8.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception as e:
        logger.warning(f"get_local_ip: could not determine local IP, defaulting to 127.0.0.1. {e}")
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def get_public_ip():
    """
    Get the public-facing IP address of this server (relevant if it's
    deployed on a cloud VM / VPS rather than run locally). Returns None if
    it can't be reached so the UI can show '--' instead of breaking.
    """
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=5)
        if r.status_code == 200:
            return r.json().get("ip")
        else:
            logger.warning(f"get_public_ip: ipify returned HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.error(f"get_public_ip: error fetching public IP: {e}")
    return None


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/signals")
def api_signals():
    try:
        market_open = is_market_open()
        logger.debug(f"api_signals: market_open={market_open}")

        candles_1m  = fetch_intraday("minutes", "1")
        candles_15m = fetch_intraday("minutes", "15")

        logger.debug(f"api_signals: raw candle counts -> 1m={len(candles_1m)}, 15m={len(candles_15m)}")

        # Reverse: API returns newest first
        candles_1m  = list(reversed(candles_1m))
        candles_15m = list(reversed(candles_15m))

        signals = generate_signals(candles_1m, candles_15m, market_open)

        # Always try to show the real, live LTP — this covers the case where the
        # market is closed (or hasn't opened yet today) and intraday candles are
        # empty, which previously caused the price to show as '--'.
        ltp = fetch_ltp()
        if ltp is not None:
            signals["indicators"]["price"] = round(ltp, 2)
        elif candles_1m:
            signals["indicators"]["price"] = round(candles_1m[-1][4], 2)
        else:
            logger.warning("api_signals: no LTP and no candles available — price will be missing/None.")

        wallet = fetch_wallet_balance()
        if wallet.get("available_margin") is None:
            logger.warning(
                "api_signals: wallet available_margin is None — check the "
                "fetch_wallet_balance logs above for the exact reason "
                "(expired token / wrong scope / API error)."
            )

        server_ip = {
            "local":  get_local_ip(),
            "public": get_public_ip()
        }

        # Latest candles for chart (last 60 1m candles)
        chart_candles = []
        for c in candles_1m[-60:]:
            chart_candles.append({
                "time":  c[0][:19].replace("T", " "),
                "open":  c[1], "high": c[2],
                "low":   c[3], "close": c[4],
                "volume": c[5]
            })

        return jsonify({
            "signals":    signals,
            "candles":    chart_candles,
            "wallet":     wallet,
            "server_ip":  server_ip,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "market_open": market_open
        })
    except Exception as e:
        logger.error(f"api_signals: unhandled exception: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/historical")
def api_historical():
    try:
        candles = fetch_historical("days", "1", 180)
        candles = list(reversed(candles))
        result = []
        for c in candles:
            result.append({
                "date":   c[0][:10],
                "open":   c[1], "high": c[2],
                "low":    c[3], "close": c[4],
                "volume": c[5]
            })
        return jsonify({"candles": result})
    except Exception as e:
        logger.error(f"api_historical: unhandled exception: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

def is_market_open():
    # IMPORTANT: always compute "now" in IST explicitly, regardless of what
    # timezone the server itself is running in. Before this fix, datetime.now()
    # used the SERVER's local clock — if the server is hosted on a machine set
    # to UTC (very common on cloud VMs/containers), the market would show as
    # CLOSED even during real IST market hours (9:15–3:30), because UTC is
    # 5 hours 30 minutes behind IST.
    now_ist = datetime.now(IST)
    logger.debug(f"is_market_open: current IST time = {now_ist.strftime('%Y-%m-%d %H:%M:%S %A')}")

    if now_ist.weekday() >= 5:  # Saturday / Sunday
        logger.info("is_market_open: weekend — market closed.")
        return False

    market_open  = now_ist.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    result = market_open <= now_ist <= market_close

    logger.info(
        f"is_market_open: now_ist={now_ist.strftime('%H:%M:%S')}, "
        f"open_window={market_open.strftime('%H:%M')}-{market_close.strftime('%H:%M')} IST, "
        f"result={result}"
    )
    return result

@app.route("/api/ws_token")
def ws_token():
    url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    logger.debug(f"ws_token: GET {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        logger.debug(f"ws_token: status_code={r.status_code}")
        body = r.json()
        if r.status_code != 200 or body.get("status") != "success":
            logger.error(f"ws_token: non-success response. HTTP={r.status_code}, body={json.dumps(body)[:1000]}")
        return jsonify(body)
    except Exception as e:
        logger.error(f"ws_token: error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/debug")
def api_debug():
    """
    Quick diagnostic endpoint. Hit this in your browser (or curl) to see
    exactly what each individual data source is returning RIGHT NOW, without
    digging through app.log. Example: http://127.0.0.1:5000/api/debug
    """
    debug_info = {}

    debug_info["access_token_present"] = bool(ACCESS_TOKEN)
    debug_info["access_token_length"]  = len(ACCESS_TOKEN) if ACCESS_TOKEN else 0
    debug_info["market_open"] = is_market_open()
    debug_info["server_raw_time"] = datetime.now().isoformat()       # whatever timezone the server/OS is in
    debug_info["ist_time"] = datetime.now(IST).isoformat()           # always actual IST, used for the market check

    # LTP
    try:
        ltp = fetch_ltp()
        debug_info["ltp"] = ltp
        debug_info["ltp_ok"] = ltp is not None
    except Exception as e:
        debug_info["ltp"] = None
        debug_info["ltp_ok"] = False
        debug_info["ltp_error"] = str(e)

    # Wallet
    try:
        wallet = fetch_wallet_balance()
        debug_info["wallet"] = wallet
        debug_info["wallet_ok"] = wallet.get("available_margin") is not None
    except Exception as e:
        debug_info["wallet"] = None
        debug_info["wallet_ok"] = False
        debug_info["wallet_error"] = str(e)

    # Intraday 1m
    try:
        c1m = fetch_intraday("minutes", "1")
        debug_info["candles_1m_count"] = len(c1m)
        debug_info["candles_1m_sample"] = c1m[:2]
    except Exception as e:
        debug_info["candles_1m_count"] = 0
        debug_info["candles_1m_error"] = str(e)

    # Intraday 15m
    try:
        c15m = fetch_intraday("minutes", "15")
        debug_info["candles_15m_count"] = len(c15m)
        debug_info["candles_15m_sample"] = c15m[:2]
    except Exception as e:
        debug_info["candles_15m_count"] = 0
        debug_info["candles_15m_error"] = str(e)

    debug_info["hint"] = (
        "If access_token_present is false -> fix your .env file. "
        "If wallet_ok is false but token is present -> check app.log for the "
        "exact HTTP status/body from get-funds-and-margin (likely 401 expired "
        "token, or 403 missing scope). "
        "If candles_1m_count is 0 during market hours -> check app.log for the "
        "fetch_intraday error (instrument key / rate limit / token issue)."
    )

    return jsonify(debug_info)


if __name__ == "__main__":
    logger.info(f"Starting Flask app — logs are printed here AND saved to {LOG_FILE}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)