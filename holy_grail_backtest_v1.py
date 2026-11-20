"""Frozen daily-bar mechanical reconstruction of Linda Raschke's Holy Grail setup.
ADX14 >30 turns upward to arm; first EMA20 pullback; DI sets direction;
entry stop one tick beyond setup bar, valid next session only; stop at setup
bar extreme; target at prior 20-bar swing; 10-bar time exit; stop-first ambiguity.
Research only. Daily Yahoo bars are not intraday fills or the trader's full discretion.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

SYMBOLS = {"QQQ": True, "SPY": True, "SOXL": False, "NVDA": False}
OUT = Path("output/holy_grail_v1")
SPLIT = "2019-01-01"
COST_SIDE_BPS = 5.0
SLIP_SIDE_BPS = 2.0
MAX_HOLD = 10


def wilder_rma(series, period=14):
    """Wilder's RMA with an SMA seed (not an unseeded EWM approximation)."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < period:
        return pd.Series(out, index=series.index)
    start = int(valid[0])
    seed_end = start + period - 1
    if seed_end >= len(values) or not np.isfinite(values[start:seed_end + 1]).all():
        return pd.Series(out, index=series.index)
    out[seed_end] = float(values[start:seed_end + 1].mean())
    for i in range(seed_end + 1, len(values)):
        if np.isfinite(values[i]):
            out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return pd.Series(out, index=series.index)


def indicators(bars):
    d = bars.copy().sort_index()
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(d.columns):
        raise ValueError(f"missing columns: {sorted(need - set(d.columns))}")
    prev = d.Close.shift(1)
    up, down = d.High.diff(), -d.Low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=d.index)
    tr = pd.concat([d.High-d.Low, (d.High-prev).abs(), (d.Low-prev).abs()], axis=1).max(axis=1)
    atr = wilder_rma(tr, 14)
    pdi = 100*wilder_rma(pdm, 14)/atr.replace(0,np.nan)
    mdi = 100*wilder_rma(mdm, 14)/atr.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d["ADX14"] = wilder_rma(dx, 14)
    d["PDI"], d["MDI"] = pdi, mdi
    d["EMA20"] = d.Close.ewm(span=20, adjust=False, min_periods=20).mean()
    d["PRIOR_HIGH20"] = d.High.shift(1).rolling(20, min_periods=20).max()
    d["PRIOR_LOW20"] = d.Low.shift(1).rolling(20, min_periods=20).min()
    return d


def setups_from_bars(bars, allow_short=True, tick=0.01):
    d, rows, armed = indicators(bars), [], None
    for i in range(40, len(d)):
        r, p = d.iloc[i], d.iloc[i-1]
        if not np.isfinite(r.ADX14) or not np.isfinite(p.ADX14):
            continue
        rising = r.ADX14 > p.ADX14 and p.ADX14 <= d.ADX14.iloc[i-2]
        if r.ADX14 > 30 and rising:
            if r.PDI > r.MDI:
                armed = "LONG"
            elif allow_short and r.MDI > r.PDI:
                armed = "SHORT"
        if armed == "LONG":
            valid = r.ADX14 > 30 and r.PDI > r.MDI
            touch = r.Low <= r.EMA20 and p.Close > p.EMA20
            if not valid:
                armed = None
            elif touch and np.isfinite(r.PRIOR_HIGH20):
                entry, stop, target = r.High+tick, r.Low-tick, r.PRIOR_HIGH20
                if stop < entry < target:
                    rows.append({"setup_date":d.index[i],"setup_index":i,"side":"LONG",
                                 "trigger":entry,"stop":stop,"target":target})
                armed = None
        elif armed == "SHORT":
            valid = r.ADX14 > 30 and r.MDI > r.PDI
            touch = r.High >= r.EMA20 and p.Close < p.EMA20
            if not valid:
                armed = None
            elif touch and np.isfinite(r.PRIOR_LOW20):
                entry, stop, target = r.Low-tick, r.High+tick, r.PRIOR_LOW20
                if target < entry < stop:
                    rows.append({"setup_date":d.index[i],"setup_index":i,"side":"SHORT",
                                 "trigger":entry,"stop":stop,"target":target})
                armed = None
    return pd.DataFrame(rows)


def simulate(bars, setups, max_hold=MAX_HOLD, cost_side_bps=COST_SIDE_BPS,
             slip_side_bps=SLIP_SIDE_BPS):
    d, dates, rows, occupied = bars.sort_index(), list(bars.sort_index().index), [], -1
    cost = 2*(cost_side_bps+slip_side_bps)/10000
    for s in setups.sort_values("setup_index").itertuples(index=False):
        si, ei = int(s.setup_index), int(s.setup_index)+1
        if ei >= len(d) or ei <= occupied:
            continue
        b = d.iloc[ei]
        if s.side == "LONG":
            filled = b.High >= s.trigger
            entry = max(float(b.Open), float(s.trigger))
            passed = b.Open >= s.target
        else:
            filled = b.Low <= s.trigger
            entry = min(float(b.Open), float(s.trigger))
            passed = b.Open <= s.target
        if not filled:
            continue
        if passed:
            xi, xp, reason = ei, entry, "target_already_passed_at_entry"
        else:
            xi = xp = reason = None
            last = min(len(dates)-1, ei+max_hold-1)
            for j in range(ei,last+1):
                bar=d.iloc[j]; o,h,l=map(float,(bar.Open,bar.High,bar.Low))
                if s.side == "LONG":
                    gap_stop, stop_hit = o<=s.stop, l<=s.stop
                    gap_target, target_hit = o>=s.target, h>=s.target
                else:
                    gap_stop, stop_hit = o>=s.stop, h>=s.stop
                    gap_target, target_hit = o<=s.target, l<=s.target
                if gap_stop or stop_hit:
                    xi,xp,reason=j,(o if gap_stop else float(s.stop)),("gap_stop" if gap_stop else "stop")
                    break
                if gap_target or target_hit:
                    xi,xp,reason=j,(o if gap_target else float(s.target)),("gap_target" if gap_target else "target")
                    break
            if xi is None:
                xi,xp,reason=last,float(d.iloc[last].Close),"time_exit"
        direction = 1 if s.side=="LONG" else -1
        gross = direction*(float(xp)-entry)/entry
        rows.append({"setup_date":s.setup_date,"entry_date":dates[ei],"exit_date":dates[xi],
                     "side":s.side,"entry":entry,"stop":s.stop,"target":s.target,"exit":float(xp),
                     "exit_reason":reason,"gross_return_pct":gross*100,
                     "cost_pct":cost*100,"net_return_pct":(gross-cost)*100,"bars_held":xi-ei+1})
        occupied=xi
    return pd.DataFrame(rows)


def stats(t):
    if t.empty:
        return {"trades":0,"win_rate_pct":None,"profit_factor":None,"avg_net_pct":None,
                "median_net_pct":None,"compounded_return_pct":0.0,"max_drawdown_pct":0.0}
    r=t.net_return_pct.astype(float)/100
    win,loss=r[r>0],r[r<0]
    eq=(1+r).cumprod()
    return {"trades":len(r),"win_rate_pct":float((r>0).mean()*100),
            "profit_factor":float(win.sum()/abs(loss.sum())) if len(loss) else None,
            "avg_net_pct":float(r.mean()*100),"median_net_pct":float(r.median()*100),
            "compounded_return_pct":float((eq.iloc[-1]-1)*100),
            "max_drawdown_pct":float((eq/eq.cummax()-1).min()*100)}


def main():
    import yfinance as yf
    OUT.mkdir(parents=True,exist_ok=True)
    summaries, ledgers=[],[]
    for symbol, short_ok in SYMBOLS.items():
        raw=yf.download(symbol,period="max",auto_adjust=True,progress=False)
        if raw is None or raw.empty:
            summaries.append({"ticker":symbol,"period":"all","status":"no_data"}); continue
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
        bars=raw[["Open","High","Low","Close","Volume"]].dropna()
        if len(bars)<250:
            summaries.append({"ticker":symbol,"period":"all","status":"insufficient_data"}); continue
        ss=setups_from_bars(bars,allow_short=short_ok)
        tr=simulate(bars,ss)
        if not tr.empty:
            tr.insert(0,"ticker",symbol); ledgers.append(tr)
        if tr.empty:
            summaries.append({"ticker":symbol,"period":"full","status":"no_trades",**stats(tr)})
        else:
            for label, mask in [("full",pd.Series(True,index=tr.index)),
                                ("in_sample_pre_2019",tr.entry_date<SPLIT),
                                ("out_of_sample_2019_plus",tr.entry_date>=SPLIT)]:
                part=tr.loc[mask]
                summaries.append({"ticker":symbol,"period":label,"status":"ok" if len(part) else "no_trades",
                                  **stats(part)})
        summaries.append({"ticker":symbol,"period":"buy_and_hold_full","status":"benchmark",
                          "buy_hold_return_pct":float((bars.Close.iloc[-1]/bars.Close.iloc[0]-1)*100)})
    ledger=pd.concat(ledgers,ignore_index=True) if ledgers else pd.DataFrame()
    summary=pd.DataFrame(summaries)
    ledger.to_csv(OUT/"trade_ledger.csv",index=False)
    summary.to_csv(OUT/"performance_summary.csv",index=False)
    method={"strategy":"Holy Grail daily mechanical reconstruction v1","symbols":SYMBOLS,
            "split":"2019-01-01","entry":"ADX14 >30 turns up; DI directional; first EMA20 touch after prior close on trend side; stop order 1 tick beyond setup-bar extreme valid next session only",
            "stop":"setup-bar extreme plus/minus one cent","target":"prior 20-session swing extreme excluding setup bar",
            "time_exit_sessions":MAX_HOLD,"cost_assumption":"5 bps each side + 2 bps slippage each side",
            "data":"Yahoo Finance adjusted daily OHLCV; no intraday or futures data",
            "limits":["not Linda's full discretionary process","daily OHLC ambiguous fills resolved stop-first",
                      "costs are assumptions, not broker fills","no parameter search; 2019+ is chronological OOS",
                      "SOXL and NVDA are long-only; QQQ/SPY long-short"]}
    (OUT/"methodology.json").write_text(json.dumps(method,indent=2),encoding="utf-8")
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(method,indent=2),flush=True)


if __name__=="__main__":
    main()
