"""
options_chain.py (v2 — FIXED)
─────────────────────────────────────────────────────────────────────────────
WHAT WAS WRONG BEFORE (why the chain showed blank):
  1. The old code manually fetched `/v2/option/contract` (just the list of
     contracts) and then tried to match each contract's instrument_key
     against `/v2/market-quote/quotes` to get LTP/OI. That match silently
     failed because of formatting differences between the two endpoints,
     so every CE/PE came back as None -> blank table.
  2. Upstox has a PURPOSE-BUILT endpoint for exactly this:
     `GET /v2/option/chain` — it requires BOTH instrument_key AND
     expiry_date, and returns strike-by-strike CE+PE data already joined,
     INCLUDING open interest, previous OI, volume, AND full option Greeks
     (delta/gamma/theta/vega/IV) in one response. No manual matching needed.
  3. `expiry_date` is a REQUIRED field on that endpoint. The old code was
     calling without ever fetching a real expiry first — Upstox was likely
     returning an error or empty data, which is the root cause of "blank".

WHAT'S NEW:
  - fetch_available_expiries(): lists every expiry currently listed for
    Nifty 50 (weekly + monthly), so the dashboard can show a dropdown —
    "nifty50 ke bahut sare chains, alag alag expiry" is handled by giving
    the user all of them, not just guessing one.
  - get_nifty_option_chain(): now hits the correct endpoint, for a given
    expiry, and returns OI, change-in-OI (oi - prev_oi), IV, and Greeks.
"""

import requests
import logging
from datetime import datetime

logger = logging.getLogger("nifty_app")

NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"


def fetch_available_expiries(access_token):
    """
    Returns a sorted list of expiry date strings (YYYY-MM-DD) currently
    available for Nifty 50 options, nearest first. Pulled from the
    contracts list (which does NOT require an expiry_date param itself).
    """
    url = "https://api.upstox.com/v2/option/contract"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    params = {"instrument_key": NIFTY_INSTRUMENT_KEY}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "success":
            logger.error(f"fetch_available_expiries: non-success. HTTP={r.status_code}, body={data}")
            return []
        contracts = data.get("data", [])
        expiries = sorted(set(c["expiry"] for c in contracts if c.get("expiry")))
        logger.info(f"fetch_available_expiries: found {len(expiries)} expiries: {expiries[:5]}...")
        return expiries
    except Exception as e:
        logger.error(f"fetch_available_expiries: error: {e}")
        return []


def get_nifty_option_chain(access_token, expiry_date=None, strikes_around_atm=10):
    """
    Fetches the live Nifty 50 option chain for a given expiry using
    Upstox's dedicated /v2/option/chain endpoint (NOT the generic quotes
    endpoint — that was the bug).

    If expiry_date is None, automatically uses the NEAREST available expiry.

    Returns:
    {
        "expiry": "2026-06-25",
        "available_expiries": ["2026-06-25", "2026-07-02", ...],
        "underlying_price": 23510.4,
        "atm_strike": 23500,
        "pcr": 1.12,
        "strikes": [
            {
                "strike": 23500,
                "CE": {"ltp":.., "oi":.., "oi_change":.., "volume":.., "iv":.., "delta":..},
                "PE": {"ltp":.., "oi":.., "oi_change":.., "volume":.., "iv":.., "delta":..}
            }, ...
        ],
        "fetched_at": "..."
    }
    """
    available_expiries = fetch_available_expiries(access_token)
    if not available_expiries:
        return {"strikes": [], "available_expiries": [], "error": "no_expiries_found"}

    if not expiry_date or expiry_date not in available_expiries:
        expiry_date = available_expiries[0]  # nearest expiry

    url = "https://api.upstox.com/v2/option/chain"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    params = {"instrument_key": NIFTY_INSTRUMENT_KEY, "expiry_date": expiry_date}

    logger.debug(f"get_nifty_option_chain: GET {url} params={params}")
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        logger.debug(f"get_nifty_option_chain: status_code={r.status_code}")
        data = r.json()
        if data.get("status") != "success":
            logger.error(f"get_nifty_option_chain: non-success. HTTP={r.status_code}, body={data}")
            return {"strikes": [], "available_expiries": available_expiries, "expiry": expiry_date, "error": "api_error"}
    except Exception as e:
        logger.error(f"get_nifty_option_chain: request error: {e}")
        return {"strikes": [], "available_expiries": available_expiries, "expiry": expiry_date, "error": str(e)}

    rows = data.get("data", [])
    if not rows:
        logger.warning(f"get_nifty_option_chain: empty data for expiry={expiry_date}")
        return {"strikes": [], "available_expiries": available_expiries, "expiry": expiry_date, "error": "empty_response"}

    underlying_price = rows[0].get("underlying_spot_price")

    def extract_side(side_obj):
        if not side_obj:
            return None
        md = side_obj.get("market_data", {}) or {}
        gk = side_obj.get("option_greeks", {}) or {}
        oi = md.get("oi")
        prev_oi = md.get("prev_oi")
        return {
            "ltp": md.get("ltp"),
            "oi": oi,
            "oi_change": (oi - prev_oi) if (oi is not None and prev_oi is not None) else None,
            "volume": md.get("volume"),
            "iv": gk.get("iv"),
            "delta": gk.get("delta"),
            "theta": gk.get("theta"),
        }

    all_strikes = []
    for row in rows:
        strike = row.get("strike_price")
        all_strikes.append({
            "strike": strike,
            "CE": extract_side(row.get("call_options")),
            "PE": extract_side(row.get("put_options")),
        })

    all_strikes.sort(key=lambda s: s["strike"])
    strike_values = [s["strike"] for s in all_strikes]

    if underlying_price and strike_values:
        atm_strike = min(strike_values, key=lambda s: abs(s - underlying_price))
    else:
        atm_strike = strike_values[len(strike_values) // 2] if strike_values else None

    if atm_strike is not None:
        atm_idx = strike_values.index(atm_strike)
        lo = max(0, atm_idx - strikes_around_atm)
        hi = min(len(strike_values), atm_idx + strikes_around_atm + 1)
        selected = all_strikes[lo:hi]
    else:
        selected = all_strikes

    total_pe_oi = sum((s["PE"]["oi"] or 0) for s in selected if s["PE"])
    total_ce_oi = sum((s["CE"]["oi"] or 0) for s in selected if s["CE"])
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi else None

    logger.info(
        f"get_nifty_option_chain: expiry={expiry_date}, strikes={len(selected)}, "
        f"atm={atm_strike}, underlying={underlying_price}, pcr={pcr}"
    )

    return {
        "expiry": expiry_date,
        "available_expiries": available_expiries,
        "underlying_price": underlying_price,
        "atm_strike": atm_strike,
        "pcr": pcr,
        "strikes": selected,
        "fetched_at": datetime.now().isoformat(),
    }