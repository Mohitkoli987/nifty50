"""
app.py — Nifty50 Dashboard (v2)
─────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM YOUR ORIGINAL FILE
  1. Signal generation now goes through signal_engine.generate_signal() —
     the SAME function backtest.py uses. No more separate "scalping /
     swing / hold" scores invented independently — one signal, confirmed
     across 1m spike + 15m trend + momentum, exactly as backtested.
  2. New /api/options route serving the live Nifty option chain (CE/PE,
     OI, PCR) via options_chain.py.
  3. All the original logging / debug / wallet / IP infrastructure is kept
     as-is — that part of your original code was solid, the problem was
     never the plumbing, it was the signal logic.

BEFORE YOU TRUST THIS LIVE: run backtest.py first. If it doesn't show a
profit factor > 1 across multiple symbols, this app's signals are
informational only — do not size real trades off them yet.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import socket
import logging
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from signal_engine import generate_signal, resample_candles
from options_chain import get_nifty_option_chain

IST = ZoneInfo("Asia/Kolkata")

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("nifty_app")

load_dotenv()

app = Flask(__name__)
CORS(app)

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
INSTRUMENT_KEY_ENCODED = "NSE_INDEX%7CNifty%2050"

if not ACCESS_TOKEN:
    logger.error("ACCESS_TOKEN is missing! Check your .env file next to app.py.")
else:
    logger.info(f"ACCESS_TOKEN loaded (length={len(ACCESS_TOKEN)} chars).")

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}


# ─── DATA FETCHING (unchanged from your original, logging intact) ────────

def fetch_intraday(unit="minutes", interval="1"):
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{INSTRUMENT_KEY_ENCODED}/{unit}/{interval}"
    logger.debug(f"fetch_intraday: GET {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        try:
            data = r.json()
        except ValueError:
            logger.error(f"fetch_intraday: invalid JSON. Raw: {r.text[:500]}")
            return []
        if data.get("status") == "success":
            candles = data["data"]["candles"]
            logger.info(f"fetch_intraday: got {len(candles)} candles for {unit}/{interval}")
            return candles
        logger.error(f"fetch_intraday: non-success. HTTP={r.status_code}, body={data}")
    except requests.exceptions.Timeout:
        logger.error(f"fetch_intraday: timeout ({url})")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"fetch_intraday: connection error: {e}")
    except Exception as e:
        logger.error(f"fetch_intraday: unexpected error: {e}\n{traceback.format_exc()}")
    return []


def fetch_ltp():
    url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={INSTRUMENT_KEY_ENCODED}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("status") == "success":
            for v in data.get("data", {}).values():
                if isinstance(v, dict) and v.get("last_price") is not None:
                    return v["last_price"]
        else:
            logger.error(f"fetch_ltp: non-success. body={data}")
    except Exception as e:
        logger.error(f"fetch_ltp: error: {e}")
    return None


def fetch_wallet_balance():
    url = "https://api.upstox.com/v2/user/get-funds-and-margin"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("status") == "success":
            equity = data.get("data", {}).get("equity", {})
            return {"available_margin": equity.get("available_margin"), "used_margin": equity.get("used_margin")}
        logger.error(f"fetch_wallet_balance: non-success. HTTP={r.status_code}, body={data}")
    except Exception as e:
        logger.error(f"fetch_wallet_balance: error: {e}")
    return {"available_margin": None, "used_margin": None}


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def get_public_ip():
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=5)
        if r.status_code == 200:
            return r.json().get("ip")
    except Exception as e:
        logger.error(f"get_public_ip: error: {e}")
    return None


def is_market_open():
    now_ist = datetime.now(IST)
    if now_ist.weekday() >= 5:
        return False
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now_ist <= market_close


# ─── ROUTES ────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/signals")
def api_signals():
    try:
        market_open = is_market_open()

        raw_1m = fetch_intraday("minutes", "1")
        raw_1m = list(reversed(raw_1m))  # ascending order for signal_engine
        candles_1m = [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in raw_1m]

        candles_15m = resample_candles(candles_1m, 15) if candles_1m else []

        sig = generate_signal(candles_1m, candles_15m, market_open)

        ltp = fetch_ltp()
        if ltp is not None:
            sig["indicators"]["price"] = round(ltp, 2)
        elif candles_1m:
            sig["indicators"]["price"] = round(candles_1m[-1][4], 2)

        wallet = fetch_wallet_balance()
        server_ip = {"local": get_local_ip(), "public": get_public_ip()}

        chart_candles = []
        for c in candles_1m[-60:]:
            chart_candles.append({
                "time": str(c[0])[:19].replace("T", " "),
                "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5],
            })

        return jsonify({
            "signal": sig,
            "candles": chart_candles,
            "wallet": wallet,
            "server_ip": server_ip,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "market_open": market_open,
        })
    except Exception as e:
        logger.error(f"api_signals: unhandled exception: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/options")
def api_options():
    """
    Live Nifty 50 option chain — CE/PE, OI, IV, Greeks, PCR, centered on
    ATM strike. Pass ?expiry=YYYY-MM-DD to pick a specific expiry; omit it
    to get the nearest expiry automatically. The response always includes
    `available_expiries` so the frontend can build a dropdown of every
    expiry currently listed for Nifty 50.

    Read the comments in options_chain.py before treating PCR or OI moves
    as a standalone trading signal — they're context, not a verdict.
    """
    try:
        expiry = request.args.get("expiry")
        chain = get_nifty_option_chain(ACCESS_TOKEN, expiry_date=expiry, strikes_around_atm=10)
        return jsonify(chain)
    except Exception as e:
        logger.error(f"api_options: unhandled exception: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug")
def api_debug():
    debug_info = {
        "access_token_present": bool(ACCESS_TOKEN),
        "market_open": is_market_open(),
        "ist_time": datetime.now(IST).isoformat(),
    }
    try:
        debug_info["ltp"] = fetch_ltp()
    except Exception as e:
        debug_info["ltp_error"] = str(e)
    try:
        debug_info["wallet"] = fetch_wallet_balance()
    except Exception as e:
        debug_info["wallet_error"] = str(e)
    try:
        c1m = fetch_intraday("minutes", "1")
        debug_info["candles_1m_count"] = len(c1m)
    except Exception as e:
        debug_info["candles_1m_error"] = str(e)
    return jsonify(debug_info)


if __name__ == "__main__":
    logger.info(f"Starting Flask app — logs printed here and saved to {LOG_FILE}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)