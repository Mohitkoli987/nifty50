"""
options_chain.py
─────────────────────────────────────────────────────────────────────────────
Fetches the live Nifty 50 options chain (CE/PE) from Upstox — strikes,
LTP, Open Interest (OI), change in OI, and Implied Volatility (IV) where
available.

WHY THIS MATTERS FOR YOUR "BUY CE/PE" GOAL
─────────────────────────────────────────────────────────────────────────────
An index-direction signal (from signal_engine.py) tells you Nifty MIGHT go
up or down. It does NOT tell you:
  - Whether buying a CALL/PUT at that moment is a good idea on its OWN
    terms — options have their own behavior layered on top of direction:
      * IV (implied volatility): if IV is already very high, premiums are
        expensive — even a correct direction call can lose money if IV
        then drops (this is called "IV crush"), especially around events.
      * Theta decay: every option loses time-value every single day,
        faster as expiry approaches. A slow, correct move can still lose
        you money if it takes too long to play out.
      * OI buildup: a sharp rise in Call OI at a strike often signals
        resistance (writers expect price to stay below it); a sharp rise
        in Put OI often signals support. This is read TOGETHER with price
        action, not alone.

This module surfaces that data so you (or a future smarter strategy) can
factor it in. It does NOT, by itself, generate a buy/sell call on options —
that requires combining this with signal_engine's direction signal AND a
view on IV/theta, which is a meaningfully harder problem than index
direction alone. Treat the numbers here as DECISION INPUT, not a signal.
"""

import requests
import logging
from datetime import datetime

logger = logging.getLogger("nifty_app")

NIFTY_KEY_ENCODED = "NSE_INDEX%7CNifty%2050"


def fetch_option_contracts(access_token, expiry_date=None):
    """
    Get the list of available option contracts (instrument keys) for
    Nifty 50, optionally filtered to a specific expiry (YYYY-MM-DD).
    If expiry_date is None, returns contracts for the NEAREST expiry.
    """
    url = "https://api.upstox.com/v2/option/contract"
    params = {"instrument_key": "NSE_INDEX|Nifty 50"}
    if expiry_date:
        params["expiry_date"] = expiry_date

    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "success":
            logger.error(f"fetch_option_contracts: non-success. HTTP={r.status_code}, body={data}")
            return []
        return data.get("data", [])
    except Exception as e:
        logger.error(f"fetch_option_contracts: error: {e}")
        return []


def fetch_option_chain_quotes(access_token, instrument_keys):
    """
    Given a list of option instrument keys (max ~50 per call — Upstox
    limits multi-quote requests), fetch live LTP/OI/volume for each.
    Returns dict keyed by instrument_key.
    """
    if not instrument_keys:
        return {}

    url = "https://api.upstox.com/v2/market-quote/quotes"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    all_data = {}

    # Upstox caps the number of instrument keys per request — chunk them.
    CHUNK = 50
    for i in range(0, len(instrument_keys), CHUNK):
        chunk = instrument_keys[i:i + CHUNK]
        params = {"instrument_key": ",".join(chunk)}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            data = r.json()
            if data.get("status") == "success":
                all_data.update(data.get("data", {}))
            else:
                logger.error(f"fetch_option_chain_quotes: non-success chunk. body={data}")
        except Exception as e:
            logger.error(f"fetch_option_chain_quotes: error on chunk: {e}")

    return all_data


def get_nifty_option_chain(access_token, expiry_date=None, strikes_around_atm=10):
    """
    High-level function: returns a clean list of dicts, one per strike,
    each containing the CE and PE side data, centered around the current
    ATM (at-the-money) strike.

    Output format per strike:
    {
        "strike": 23500,
        "CE": {"ltp": .., "oi": .., "oi_change": .., "volume": ..},
        "PE": {"ltp": .., "oi": .., "oi_change": .., "volume": ..}
    }
    """
    contracts = fetch_option_contracts(access_token, expiry_date)
    if not contracts:
        return {"expiry": expiry_date, "strikes": [], "error": "no_contracts_found"}

    # contracts come with instrument_key, strike_price, instrument_type (CE/PE), expiry
    instrument_keys = [c["instrument_key"] for c in contracts]
    quotes = fetch_option_chain_quotes(access_token, instrument_keys)

    # group by strike
    by_strike = {}
    for c in contracts:
        strike = c.get("strike_price")
        opt_type = c.get("instrument_type")  # "CE" or "PE"
        ikey = c.get("instrument_key")
        q = quotes.get(ikey, {}) or {}

        if strike not in by_strike:
            by_strike[strike] = {"strike": strike, "CE": None, "PE": None}

        by_strike[strike][opt_type] = {
            "ltp": q.get("last_price"),
            "oi": q.get("oi"),
            "volume": q.get("volume"),
            "instrument_key": ikey,
        }

    sorted_strikes = sorted(by_strike.keys())

    # Determine ATM strike using the underlying's LTP if we can get it,
    # otherwise fall back to the median strike in the contract list.
    try:
        underlying_resp = requests.get(
            f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={NIFTY_KEY_ENCODED}",
            headers={"Accept": "application/json", "Authorization": f"Bearer {access_token}"},
            timeout=10,
        ).json()
        underlying_price = None
        for v in (underlying_resp.get("data", {}) or {}).values():
            if "last_price" in v:
                underlying_price = v["last_price"]
                break
    except Exception as e:
        logger.warning(f"get_nifty_option_chain: could not fetch underlying LTP: {e}")
        underlying_price = None

    if underlying_price:
        atm_strike = min(sorted_strikes, key=lambda s: abs(s - underlying_price))
    else:
        atm_strike = sorted_strikes[len(sorted_strikes) // 2] if sorted_strikes else None

    if atm_strike is None:
        return {"expiry": expiry_date, "strikes": [], "error": "no_strikes"}

    atm_idx = sorted_strikes.index(atm_strike)
    lo = max(0, atm_idx - strikes_around_atm)
    hi = min(len(sorted_strikes), atm_idx + strikes_around_atm + 1)
    selected_strikes = sorted_strikes[lo:hi]

    result_strikes = [by_strike[s] for s in selected_strikes]

    # Simple PCR (Put-Call Ratio) on OI — widely used as a sentiment gauge:
    # PCR > 1 generally read as more bullish positioning (more puts being
    # written/sold relative to calls), PCR < 1 more bearish. Read this as
    # one input among several, not a standalone signal.
    total_pe_oi = sum((s["PE"]["oi"] or 0) for s in result_strikes if s["PE"])
    total_ce_oi = sum((s["CE"]["oi"] or 0) for s in result_strikes if s["CE"])
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi else None

    return {
        "expiry": expiry_date or (contracts[0].get("expiry") if contracts else None),
        "underlying_price": underlying_price,
        "atm_strike": atm_strike,
        "pcr": pcr,
        "strikes": result_strikes,
        "fetched_at": datetime.now().isoformat(),
    }