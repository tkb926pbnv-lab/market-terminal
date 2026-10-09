"""Market Terminal core: real data (Yahoo Finance via yfinance), indicators, signals, 'errors' scanner.
NOT FINANCIAL ADVICE. Educational/informational tool only."""
import time, math, datetime as dt
import numpy as np, pandas as pd, yfinance as yf

GROUPS = [
    {"name": "S&P 500", "index": "^GSPC", "etf": "SPY", "ref": "^GSPC"},
    {"name": "Nasdaq", "index": "^IXIC", "etf": "QQQ", "ref": "^NDX"},   # QQQ tracks Nasdaq-100, so compare vs ^NDX
    {"name": "Dow Jones", "index": "^DJI", "etf": "DIA", "ref": "^DJI"},
]
TICKERS = ["^GSPC", "SPY", "^IXIC", "QQQ", "^NDX", "^DJI", "DIA"]
DISCLAIMER = ("NOT FINANCIAL ADVICE. For informational/educational purposes only. Signals are mechanical "
              "technical-analysis outputs, can be wrong, and past behavior does not predict future results. "
              "Data from Yahoo Finance may be delayed (indices up to ~15 min) or incomplete. Trade at your own risk.")

_cache = {"t": 0, "data": None}

def _clean(x):
    if x is None: return None
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else round(float(x), 4)
    if isinstance(x, (np.integer,)): return int(x)
    return x

def _hist(period, interval):
    df = yf.download(TICKERS, period=period, interval=interval, auto_adjust=False,
                     progress=False, group_by="ticker", threads=True)
    return df

def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean(); ad = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + au / ad)

def macd(s):
    m = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig

def swing_levels(df, lookback=120, w=5):
    d = df.tail(lookback)
    hi, lo = d["High"], d["Low"]
    res = [hi.iloc[i] for i in range(w, len(d)-w) if hi.iloc[i] == hi.iloc[i-w:i+w+1].max()]
    sup = [lo.iloc[i] for i in range(w, len(d)-w) if lo.iloc[i] == lo.iloc[i-w:i+w+1].min()]
    return sorted(set(map(float, sup))), sorted(set(map(float, res)))

def analyze(sym, daily, intra, live_price=None):
    df = daily[sym].dropna(subset=["Close"]).copy()
    c = df["Close"]
    last = float(live_price) if live_price else float(c.iloc[-1])
    prev = float(c.iloc[-2]) if df.index[-1].date() == dt.date.today() or len(c) > 1 else float(c.iloc[-1])
    # previous close = last completed session
    today = df.index[-1].date()
    prev_close = float(c.iloc[-2])
    sma20, sma50, sma200 = (c.rolling(n).mean() for n in (20, 50, 200))
    std20 = c.rolling(20).std()
    bb_up, bb_lo = sma20 + 2*std20, sma20 - 2*std20
    r = rsi(c); m, ms, mh = macd(c)
    z = (last - sma20.iloc[-1]) / std20.iloc[-1]
    pctb = (last - bb_lo.iloc[-1]) / (bb_up.iloc[-1] - bb_lo.iloc[-1])
    # classic floor pivots from previous completed session
    p = df.iloc[-2]; P = (p.High + p.Low + p.Close)/3
    piv = {"P": P, "R1": 2*P - p.Low, "S1": 2*P - p.High, "R2": P + (p.High - p.Low), "S2": P - (p.High - p.Low)}
    sups, ress = swing_levels(df)
    cand_sup = sorted([x for x in sups + [piv["S1"], piv["S2"], bb_lo.iloc[-1], sma50.iloc[-1], sma200.iloc[-1]] if x < last], reverse=True)
    cand_res = sorted([x for x in ress + [piv["R1"], piv["R2"], bb_up.iloc[-1]] if x > last])
    s1 = cand_sup[0] if cand_sup else bb_lo.iloc[-1]
    s2 = cand_sup[1] if len(cand_sup) > 1 else s1 * 0.99
    r1 = cand_res[0] if cand_res else bb_up.iloc[-1]
    r2 = cand_res[1] if len(cand_res) > 1 else r1 * 1.01
    # buy zone: between lower Bollinger band / nearest strong support; sell zone: upper band / nearest resistance
    buy_zone = sorted([float(min(s1, bb_lo.iloc[-1])), float(max(s1, bb_lo.iloc[-1]))]) if bb_lo.iloc[-1] < last else [float(s2), float(s1)]
    sell_zone = sorted([float(min(r1, bb_up.iloc[-1])), float(max(r1, bb_up.iloc[-1]))]) if bb_up.iloc[-1] > last else [float(r1), float(r2)]
    # composite score
    score, why = 0, []
    def add(cond, pts, txt):
        nonlocal score
        if cond: score += pts; why.append(txt)
    add(last > sma200.iloc[-1], 1, "above 200-SMA (long-term uptrend)")
    add(last < sma200.iloc[-1], -1, "below 200-SMA (long-term downtrend)")
    add(sma50.iloc[-1] > sma200.iloc[-1], 1, "50-SMA > 200-SMA (golden-cross regime)")
    add(sma50.iloc[-1] < sma200.iloc[-1], -1, "50-SMA < 200-SMA (death-cross regime)")
    add(last > sma20.iloc[-1], 0.5, "above 20-SMA (short-term up)")
    add(last < sma20.iloc[-1], -0.5, "below 20-SMA (short-term down)")
    add(mh.iloc[-1] > 0, 0.5, "MACD above signal (bullish momentum)")
    add(mh.iloc[-1] < 0, -0.5, "MACD below signal (bearish momentum)")
    add(r.iloc[-1] > 70, -1, f"RSI {r.iloc[-1]:.0f} overbought")
    add(r.iloc[-1] < 30, 1, f"RSI {r.iloc[-1]:.0f} oversold")
    add(pctb > 1, -1, "price above upper Bollinger band (stretched)")
    add(pctb < 0, 1, "price below lower Bollinger band (stretched)")
    bias = "BULLISH" if score >= 1.5 else "BEARISH" if score <= -1.5 else "NEUTRAL"
    if pctb < 0.15 and bias != "BEARISH": action = "BUY ZONE – price near support in an intact trend"
    elif pctb > 0.85 and bias != "BULLISH": action = "SELL/TRIM ZONE – stretched toward resistance"
    elif pctb > 0.95: action = "HOLD / DON'T CHASE – extended at upper band; wait for pullback to buy zone"
    elif pctb < 0.05: action = "WAIT – falling below lower band in weak trend; let it stabilize"
    else: action = "WAIT – mid-range; best entries are in the buy zone, exits in the sell zone"
    explain = (f"{bias} bias (score {score:+.1f}): " + "; ".join(why) + ". "
               f"Buy zone {buy_zone[0]:,.2f}–{buy_zone[1]:,.2f} = nearest support / lower Bollinger band "
               f"({(buy_zone[1]/last-1)*100:+.2f}% from here). Sell zone {sell_zone[0]:,.2f}–{sell_zone[1]:,.2f} = "
               f"nearest resistance / upper band ({(sell_zone[0]/last-1)*100:+.2f}%).")
    # intraday
    it = intra[sym].dropna(subset=["Close"]) if sym in intra.columns.get_level_values(0) else pd.DataFrame()
    it = it[it.index.date == it.index[-1].date()] if len(it) else it
    o = float(df["Open"].iloc[-1]) if today == (it.index[-1].date() if len(it) else today) else None
    hist = df.tail(250)
    return {
        "symbol": sym, "price": last, "prev_close": prev_close, "change": last - prev_close,
        "change_pct": (last/prev_close - 1)*100, "open": o,
        "day_high": float(df["High"].iloc[-1]), "day_low": float(df["Low"].iloc[-1]),
        "asof": (it.index[-1] if len(it) else df.index[-1]).isoformat(),
        "sma20": sma20.iloc[-1], "sma50": sma50.iloc[-1], "sma200": sma200.iloc[-1],
        "rsi": r.iloc[-1], "macd": m.iloc[-1], "macd_signal": ms.iloc[-1], "macd_hist": mh.iloc[-1],
        "bb_upper": bb_up.iloc[-1], "bb_lower": bb_lo.iloc[-1], "pct_b": pctb, "zscore20": z,
        "pivots": {k: float(v) for k, v in piv.items()},
        "support": [float(s1), float(s2)], "resistance": [float(r1), float(r2)],
        "buy_zone": buy_zone, "sell_zone": sell_zone, "bias": bias, "score": score,
        "action": action, "explain": explain,
        "chart_daily": {"t": [d.strftime("%Y-%m-%d") for d in hist.index], "c": [float(x) for x in hist.Close],
                         "sma20": [_clean(x) for x in sma20.tail(250)], "sma50": [_clean(x) for x in sma50.tail(250)],
                         "sma200": [_clean(x) for x in sma200.tail(250)],
                         "bbu": [_clean(x) for x in bb_up.tail(250)], "bbl": [_clean(x) for x in bb_lo.tail(250)]},
        "chart_intra": {"t": [d.strftime("%H:%M") for d in it.index], "c": [float(x) for x in it.Close]} if len(it) else {"t": [], "c": []},
    }

def errors_scan(res, daily):
    flags = []
    # 1) ETF vs reference index divergence: ratio today vs its 60-day mean (unadjusted closes)
    for g in GROUPS:
        e, ref = g["etf"], g["ref"]
        a, b = daily[e]["Close"].dropna(), daily[ref]["Close"].dropna()
        j = pd.concat([a, b], axis=1, keys=["e", "i"]).dropna()
        ratio = (j.e / j.i)
        hist = ratio.iloc[-61:-1]
        cur = res[e]["price"] / res[ref]["price"]
        dev_bp = (cur / hist.mean() - 1) * 1e4
        zz = (cur - hist.mean()) / hist.std()
        sev = "HIGH" if abs(zz) > 3 else "MED" if abs(zz) > 2 else "LOW"
        flags.append({"type": "ETF/Index divergence", "symbol": f"{e} vs {ref}", "severity": sev,
            "value": f"{dev_bp:+.1f} bp vs 60d avg ratio (z={zz:+.2f})",
            "note": (f"{e} trading {'rich (premium)' if dev_bp>0 else 'cheap (discount)'} vs {ref}. "
                     + ("Unusual - potential relative-value/mean-reversion setup; verify it's not a dividend ex-date or stale index print." if sev != "LOW"
                        else "Within normal range - no actionable mispricing."))})
    # 2) Overnight gaps
    for s in TICKERS:
        if s == "^NDX": continue
        r = res[s]
        if r["open"] is None: continue
        gap = (r["open"] / r["prev_close"] - 1) * 100
        filled = (r["day_low"] <= r["prev_close"] <= r["day_high"])
        sev = "HIGH" if abs(gap) > 1 else "MED" if abs(gap) > 0.4 else "LOW"
        flags.append({"type": "Overnight gap", "symbol": s, "severity": sev,
            "value": f"{gap:+.2f}% (open {r['open']:,.2f} vs prev close {r['prev_close']:,.2f})",
            "note": ("Gap already FILLED today." if filled else
                     f"Gap UNFILLED - gap-fill target {r['prev_close']:,.2f} ({(r['prev_close']/r['price']-1)*100:+.2f}% from last).")
                    if sev != "LOW" else "Small gap - noise."})
    # 3) Extreme deviations from mean / Bollinger
    for s in TICKERS:
        r = res[s]; zz = r["zscore20"]
        sev = "HIGH" if abs(zz) > 2.5 else "MED" if abs(zz) > 2 else "WATCH" if abs(zz) > 1.5 else "LOW"
        flags.append({"type": "Mean-reversion", "symbol": s, "severity": sev,
            "value": f"z={zz:+.2f} vs 20-SMA, %B={r['pct_b']:.2f}, RSI={r['rsi']:.1f}",
            "note": (f"Stretched {'above' if zz>0 else 'below'} mean - reversion target 20-SMA {r['sma20']:,.2f} "
                     f"({(r['sma20']/r['price']-1)*100:+.2f}%)." if sev in ("HIGH", "MED", "WATCH") else "Near mean - no edge.")})
    order = {"HIGH": 0, "MED": 1, "WATCH": 2, "LOW": 3}
    return sorted(flags, key=lambda f: order[f["severity"]])

def get_data(max_age=60):
    if _cache["data"] and time.time() - _cache["t"] < max_age:
        return _cache["data"]
    daily = _hist("2y", "1d")
    intra = _hist("5d", "5m")
    res = {}
    for s in TICKERS:
        lp = None
        try:
            ic = intra[s]["Close"].dropna()
            if len(ic) and ic.index[-1].date() == daily[s].dropna(subset=["Close"]).index[-1].date():
                lp = float(ic.iloc[-1])
        except Exception: pass
        res[s] = analyze(s, daily, intra, lp)
    out = {"generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "groups": GROUPS, "data": res, "errors": errors_scan(res, daily), "disclaimer": DISCLAIMER}
    out = _deep_clean(out)
    _cache.update(t=time.time(), data=out)
    return out

def _deep_clean(o):
    if isinstance(o, dict): return {k: _deep_clean(v) for k, v in o.items()}
    if isinstance(o, list): return [_deep_clean(v) for v in o]
    return _clean(o)
