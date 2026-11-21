"""V15: monthly walk-forward QQQ 5-session direction model.

Features are price/volume only. At each month start, fit on the trailing
5 years of matured labels; labels use the next 5 trading sessions and are
excluded until fully observable. Predictions are then traded next close.
Cash-only, whole shares, long-only, no leverage/shorts.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, precision_score, recall_score

INITIAL_CASH = 10_000.0
HORIZON = 5
TRAIN_WINDOW = 1260
MIN_TRAIN = 756
COSTS = [10.0, 20.0]
OUT = Path("output/qqq_v15_walkforward_ml")
OUT.mkdir(parents=True, exist_ok=True)

def download_bars():
    raw = yf.download("QQQ", period="max", interval="1d", auto_adjust=False,
                      actions=True, progress=False, threads=False, timeout=30)
    if raw is None or raw.empty:
        raise RuntimeError("QQQ daily data unavailable")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Missing required fields: {missing}")
    for c in ["Dividends", "Stock Splits"]:
        if c not in raw.columns:
            raw[c] = 0.0
    raw = raw[required + ["Dividends", "Stock Splits"]].apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw.index = idx.normalize()
    return raw.dropna(subset=["Open","High","Low","Close","Volume"]).loc[lambda x: ~x.index.duplicated(keep="last")].sort_index()

def rsi(s, n):
    d=s.diff()
    up=d.clip(lower=0).rolling(n).mean()
    dn=(-d.clip(upper=0)).rolling(n).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def build_features(b):
    c,h,l,o,v=b["Close"],b["High"],b["Low"],b["Open"],b["Volume"]
    ret=c.pct_change()
    f=pd.DataFrame(index=b.index)
    for n in [1,3,5,10,20]:
        f[f"ret_{n}"]=c.pct_change(n)
    for n in [20,50,100,200]:
        ma=c.rolling(n).mean()
        f[f"dist_sma_{n}"]=c/ma-1
    f["vol_5"]=ret.rolling(5).std()*np.sqrt(252)
    f["vol_20"]=ret.rolling(20).std()*np.sqrt(252)
    f["vol_60"]=ret.rolling(60).std()*np.sqrt(252)
    f["rsi_2"]=rsi(c,2)/100
    f["rsi_14"]=rsi(c,14)/100
    prev=c.shift(1)
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    f["atr14_pct"]=tr.rolling(14).mean()/c
    f["range_pct"]=(h-l)/c
    f["overnight_gap"] = o/prev-1
    f["close_location"]=(c-l)/(h-l).replace(0,np.nan)
    vmean=v.rolling(20).mean()
    f["volume_ratio"]=v/vmean.replace(0,np.nan)
    f["volume_change_5"]=v.pct_change(5)
    f=f.replace([np.inf,-np.inf],np.nan)
    return f

def build_labels(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Five-session return label from the next-close entry to its exit.

    A label is valid only when both the next-session entry close and the
    horizon exit close are observable. This masks the full horizon+1 tail.
    """
    entry_close = close.shift(-1)
    exit_close = close.shift(-(horizon + 1))
    label = (exit_close / entry_close - 1.0 > 0).astype(float)
    return label.where(entry_close.notna() & exit_close.notna())


def walk_forward_predictions(bars, features):
    close=bars["Close"]
    labels=build_labels(close)
    pred=pd.Series(np.nan,index=bars.index,dtype=float)
    fitted=0
    months=pd.Series(bars.index.to_period("M"),index=bars.index)
    for _, group in months.groupby(months).groups.items():
        pos=bars.index.get_indexer(group)
        first=int(pos[0])
        cutoff=first-(HORIZON+1)
        if cutoff<0:
            continue
        train_end=bars.index[cutoff]
        train_mask=(bars.index<=train_end)&labels.notna()&features.notna().all(axis=1)
        train_idx=np.flatnonzero(train_mask.to_numpy())[-TRAIN_WINDOW:]
        if len(train_idx)<MIN_TRAIN:
            continue
        model=Pipeline([
            ("scale",StandardScaler()),
            ("logit",LogisticRegression(C=0.2,solver="lbfgs",max_iter=500))
        ])
        X=features.iloc[train_idx]
        y=labels.iloc[train_idx].astype(int)
        if y.nunique()<2:
            continue
        model.fit(X,y)
        valid_idx=[i for i in pos if features.iloc[i].notna().all()]
        if valid_idx:
            pred.iloc[valid_idx]=model.predict_proba(features.iloc[valid_idx])[:,1]
            fitted+=1
    return pred,fitted,labels

def simulate(name,bars,target,cost_bps,buy_at_start=False):
    cash,shares,costs,dividends=INITIAL_CASH,0,0.0,0.0
    weights=target.reindex(bars.index).fillna(0).clip(0,1).shift(1).fillna(0)
    if buy_at_start: weights.iloc[0]=1.0
    prior=-1.0 if buy_at_start else 0.0
    rows=[]; trades=[]; peak=INITIAL_CASH
    for i,(date,row) in enumerate(bars.iterrows()):
        price=float(row.Close)
        div=float(row.Dividends) if pd.notna(row.Dividends) else 0.0
        if div>0 and shares:
            amount=shares*div; cash+=amount; dividends+=amount
        w=float(weights.iloc[i])
        if abs(w-prior)>1e-10:
            equity=cash+shares*price
            desired=int((equity*w)//price) if w>0 else 0
            if desired>shares:
                qty=min(desired-shares,int(cash//(price*(1+cost_bps/10000))))
                if qty>0:
                    notional=qty*price; fee=notional*cost_bps/10000
                    cash-=notional+fee; shares+=qty; costs+=fee
                    trades.append((date,"BUY",qty,price,fee))
            elif desired<shares:
                qty=shares-desired; notional=qty*price; fee=notional*cost_bps/10000
                cash+=notional-fee; shares-=qty; costs+=fee
                trades.append((date,"SELL",qty,price,fee))
            prior=w
        eq=cash+shares*price; peak=max(peak,eq)
        rows.append({"date":str(date.date()),"equity":eq,"cash":cash,"shares":shares,
                     "market_value":shares*price,"invested_pct":shares*price/eq*100 if eq else 0,
                     "drawdown_pct":(eq/peak-1)*100})
    curve=pd.DataFrame(rows)
    daily=curve.equity.pct_change().dropna()
    years=max((pd.to_datetime(curve.date.iloc[-1])-pd.to_datetime(curve.date.iloc[0])).days/365.25,1/365.25)
    end=float(curve.equity.iloc[-1])
    summary={"strategy":name,"cost_bps_per_order":cost_bps,"ending_equity_usd":end,
      "total_return_pct":(end/INITIAL_CASH-1)*100,
      "cagr_pct":((end/INITIAL_CASH)**(1/years)-1)*100 if end>0 else None,
      "max_drawdown_pct":float((curve.equity/curve.equity.cummax()-1).min()*100),
      "sharpe_rf0":float(daily.mean()/daily.std(ddof=1)*np.sqrt(252)) if len(daily)>1 and daily.std(ddof=1)>0 else None,
      "orders":len(trades),"trading_cost_usd":costs,"cash_dividends_usd":dividends,
      "average_invested_pct":float(curve.invested_pct.mean()),
      "start_date":curve.date.iloc[0],"end_date":curve.date.iloc[-1]}
    ledger=pd.DataFrame(trades,columns=["date","side","shares","price","cost"])
    return summary,curve,ledger

def main():
    bars=download_bars()
    features=build_features(bars)
    prob,fitted,labels=walk_forward_predictions(bars,features)
    trend=((bars.Close>bars.Close.rolling(200).mean()) &
           (bars.Close.rolling(50).mean()>bars.Close.rolling(200).mean())).astype(float)
    strategies={"QQQ_buy_and_hold":(pd.Series(1.0,index=bars.index),True),
                "trend_filter_control":(trend,False)}
    for threshold in [0.50,0.55,0.60]:
        strategies[f"walkforward_logit_p{threshold:.2f}"]=((prob>threshold).astype(float),False)
    summaries=[]; curves=[]; ledgers=[]
    for cost in COSTS:
        for name,(target,bh) in strategies.items():
            s,c,l=simulate(name,bars,target,cost,bh)
            summaries.append(s); c.insert(1,"strategy",name); curves.append(c)
            if len(l): l.insert(1,"strategy",name); ledgers.append(l)
    summary_df=pd.DataFrame(summaries)
    summary_df.to_csv(OUT/"account_summary.csv",index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUT/"equity_curves.csv",index=False)
    if ledgers: pd.concat(ledgers,ignore_index=True).to_csv(OUT/"trade_ledger.csv",index=False)
    pred_df=pd.DataFrame({"date":bars.index.astype(str),"prob_up_5d":prob.values,
                          "label_up_5d":labels.values})
    pred_df.to_csv(OUT/"walkforward_predictions.csv",index=False)
    valid=prob.notna()&labels.notna()
    metrics={}
    if valid.sum()>20 and labels[valid].nunique()>1:
        y=labels[valid].astype(int); p=prob[valid]
        metrics={"n_predictions":int(valid.sum()),"roc_auc":float(roc_auc_score(y,p)),
          "accuracy_at_50":float(accuracy_score(y,(p>0.5).astype(int))),
          "brier_score":float(brier_score_loss(y,p)),
          "precision_at_55":float(precision_score(y,(p>0.55).astype(int),zero_division=0)),
          "recall_at_55":float(recall_score(y,(p>0.55).astype(int),zero_division=0))}
    meta={"status":"completed","ticker":"QQQ","initial_cash_usd":INITIAL_CASH,
      "prediction_horizon_sessions":HORIZON,"retraining":"Monthly expanding calendar with trailing 1260 matured observations; minimum 756.",
      "label":"Close(t+6)/Close(t+1)-1 > 0, matching next-close entry and a 5-session holding horizon; labels included only when fully matured by month start.",
      "execution":"Signal after close t; target position executed at close t+1.",
      "model":"StandardScaler + LogisticRegression(C=0.2), fixed parameters; no hyperparameter tuning.",
      "features":list(features.columns),"model_refits":fitted,"prediction_metrics":metrics,
      "cost_grid_bps_per_order":COSTS,
      "account":"Whole shares, cash only, long only, no leverage, no shorting.",
      "dividends":"Cash distributions credited to shares held at prior close; not reinvested.",
      "limitations":["Price/volume only; no news or macro sentiment features yet.",
        "Historical walk-forward is not a guarantee of future performance.",
        "Close fills approximate execution; spread, gaps, intraday fills and market impact excluded.",
        "Cash interest, taxes and dividend reinvestment excluded.",
        "Only QQQ tested; no whole-market stock selection."]}
    (OUT/"experiment_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(summary_df[["strategy","cost_bps_per_order","ending_equity_usd","cagr_pct",
      "max_drawdown_pct","sharpe_rf0","orders","trading_cost_usd","average_invested_pct"]].to_string(index=False),flush=True)
    print("WALK_FORWARD_METRICS",json.dumps(metrics),flush=True)

if __name__=="__main__":
    main()
