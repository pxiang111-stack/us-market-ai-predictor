#!/usr/bin/env python3
"""Poll free Alpaca IEX snapshots; invoke DeepSeek only on material moves or stale report."""
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from alpaca_market_data import get_free_quote_snapshot

ROOT=Path(__file__).resolve().parent
WATCHLIST=["NVDA","MSFT","AAPL","META","AMZN","TSLA","QQQ","SPY"]
MOVE_THRESHOLD_PCT=0.75
MAX_COMMITTEE_AGE_MIN=60

def utc_now(): return datetime.now(timezone.utc)

def main():
    snapshot=get_free_quote_snapshot(WATCHLIST)
    snapshot["checked_at_utc"]=utc_now().isoformat()
    snapshot["change_vs_last_committee_pct"]={}
    snapshot["committee_trigger_reason"]=None
    (ROOT/"intraday_snapshot.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    quotes=snapshot.get("quotes",{})
    missing=[s for s in WATCHLIST if not quotes.get(s,{}).get("fresh")]
    if missing:
        print(f"Alpaca免费IEX行情缺失或超过10分钟：{missing}；跳过委员会。")
        return
    report_path=ROOT/"daily_report.json"
    old={}
    if report_path.exists():
        try: old=json.loads(report_path.read_text(encoding="utf-8"))
        except Exception: old={}
    old_prices={x.get("ticker"):x.get("market",{}).get("current_price") for x in old.get("results",[])}
    age_min=None
    try:
        generated=datetime.fromisoformat(old["generated_at_utc"].replace("Z","+00:00"))
        age_min=(utc_now()-generated).total_seconds()/60
    except Exception: pass
    moves={}
    for symbol,q in quotes.items():
        previous=old_prices.get(symbol)
        if isinstance(previous,(int,float)) and previous>0:
            moves[symbol]=round((q["price"]/previous-1)*100,3)
    max_move=max((abs(v) for v in moves.values()),default=0)
    reason=None
    if len(old_prices)!=len(WATCHLIST): reason="首次运行或缺少完整委员会基准报告"
    elif max_move>=MOVE_THRESHOLD_PCT: reason=f"至少一只标的相对上次委员会基准异动达到{max_move:.3f}%"
    elif age_min is None or age_min>=MAX_COMMITTEE_AGE_MIN:
        reason=f"距离上次委员会分析已{round(age_min,1) if age_min is not None else '未知'}分钟"
    snapshot["change_vs_last_committee_pct"]=moves
    snapshot["committee_trigger_reason"]=reason
    (ROOT/"intraday_snapshot.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if reason:
        print("触发DeepSeek委员会："+reason)
        result=subprocess.run([sys.executable,str(ROOT/"quant_committee.py")],cwd=str(ROOT))
        if result.returncode: raise SystemExit(result.returncode)
    else: print(f"Alpaca免费IEX正常；最大异动{max_move:.3f}%，委员会报告年龄{age_min:.1f}分钟，暂不调用DeepSeek。")

if __name__=="__main__": main()
