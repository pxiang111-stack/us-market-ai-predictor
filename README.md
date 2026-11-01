# US Market AI Opportunity Scanner — V2

## Purpose
Daily research scanner for liquid US equities. Its primary output ranks stocks by a six-month (126 trading-day) opportunity score. It also produces 1-month and 3-month estimates for context.

## Universe
Current V2 starts with S&P 500, Nasdaq-100, Dow 30 constituents and selected market/sector ETFs. The six-month stock leaderboard excludes the ETF list. This is a practical liquid-stock universe, **not every US-listed security**. The constituent lists are current lists, so historical tests can contain survivorship bias.

## Install and run
Python 3.10+:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python us_market_ai_v2.py
```
Windows activation:
```powershell
.venv\Scripts\Activate.ps1
```

Outputs:
- `output/top_6m_opportunities.csv`: primary six-month stock leaderboard.
- `output/all_predictions.csv`: predictions for all tickers and horizons.
- `output/validation_report.csv`: chronological 80/20 holdout metrics.

## Ranking logic
- Direction: blend of Logistic Regression and Random Forest probability estimates.
- Return: HistGradientBoosting regression estimate.
- Score: combines probability and estimated return, then penalizes high realized volatility and deep drawdown.
- The score is a ranking heuristic, not a calibrated probability or a promised return. Compare candidates with validation metrics and fundamentals before acting.

## News sentiment input
V2 supports optional `news_sentiment.csv`:
```csv
date,ticker,sentiment
2026-10-30,NVDA,0.55
2026-10-30,MARKET,-0.10
```
Sentiment values must be in [-1,1] and must reflect information publicly available at the timestamp. V2 shifts sentiment one trading day to reduce look-ahead risk. The code does not itself fetch, verify, or classify live news. If no file exists, sentiment defaults to neutral zero. Do not mistake this for a fully connected news feed.

## Daily scheduling
The included GitHub Actions workflow runs on weekdays after the US close and stores the CSV outputs as downloadable workflow artifacts. To activate:
1. Create a private GitHub repository.
2. Upload this project, including `.github/workflows/daily_scan.yml`.
3. In GitHub repository Settings > Actions > General, allow Actions if disabled.
4. The workflow runs on its schedule; you can also use Actions > Daily US Stock Scan > Run workflow.
GitHub cron uses UTC and may run later than scheduled. The workflow is not guaranteed to run on market holidays only; it skips weekends but may run on holidays.

For a local daily run, use Windows Task Scheduler or macOS/Linux cron to run `python us_market_ai_v2.py` after the US market close.

## Critical limitations before real-money use
- Free Yahoo Finance data may be incomplete, delayed, revised, throttled or unavailable.
- Current index constituents create survivorship bias in historical validation; use point-in-time constituents and delisted names for institutional-quality testing.
- The simple 80/20 holdout is preliminary. Add rolling-origin folds, purging/embargo for overlapping forward-return labels, calibration, benchmark comparisons, transaction costs and paper trading.
- News data is only an optional manual CSV. Live timestamped news ingestion and finance-specific NLP are a separate integration.
- No fundamental financial statement features yet; add point-in-time earnings, revenue growth, margins, valuation and balance-sheet measures before using the leaderboard as an investment shortlist.
- Do not buy solely because of a high rank or probability. This is a research tool, not personalized financial advice.
