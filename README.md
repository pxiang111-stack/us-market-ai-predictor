# US Alpha Research System — V4 research upgrade

Research prototype for ranking a broad universe of US equities for 1-, 3- and 6-month horizons. V4 adds a transparent cross-sectional factor engine and periodic long-only factor portfolio evaluation.

## Outputs
- `output/top_25_6m.csv`: top 25 six-month stock candidates
- `output/top_6m_opportunities.csv`: full six-month stock ranking
- `output/all_predictions.csv`: predictions for all horizons
- `output/validation_report.csv`: time-based holdout validation metrics, including top-decile return and hit-rate diagnostics
- `output/run_summary.csv`: coverage and run metadata
- `output/news_sentiment_report.csv`: recent headline sentiment and article counts
- `output/factor_scores.csv`: date-wise cross-sectional factor scores and component ranks
- `output/factor_validation_report.csv`: factor rank-IC and final-30%-period portfolio evaluation summaries
- `output/factor_backtest_detail.csv`: rebalance-level factor basket returns, estimated costs, turnover proxy and SPY comparison

## Limitations
This is not investment advice and has no guaranteed predictive edge. The universe is not literally every US-listed security; screeners, Yahoo Finance coverage, rate limits and missing data can reduce coverage. Survivorship bias remains because historical constituents and delisted securities are not reconstructed. The model includes a lagged `news_sentiment` feature when a historical `news_sentiment.csv` file is supplied (columns: `date,ticker,sentiment`; sentiment range -1 to 1). Without that file, the feature is missing and is imputed with a missingness indicator; it should not be interpreted as having contributed predictive information. The V3.1 news overlay separately fetches recent headlines from the free GDELT DOC API (no API key) and scores headline tone with VADER. It checks the broad market plus up to 40 leading six-month candidates. Coverage and ticker ambiguity vary; headline polarity is not a substitute for reading the articles. This news overlay is recent-only and is not included in the historical backtest, so its contribution is not yet historically validated. The V4 factor score is a transparent cross-sectional heuristic built from price/volume-derived momentum, relative strength, trend, low-volatility, drawdown and RSI-balance factors; it does not yet include point-in-time fundamentals. Factor backtests use non-overlapping horizon-length baskets over the final 30% of available dates and an estimated turnover-based transaction-cost proxy. They are not a full institutional simulator and do not remove survivorship bias. Scores are sorting heuristics, not calibrated probabilities of profit or expected returns. Treat AUC near 0.50, rank IC near zero, or non-positive net excess returns as no demonstrated predictive edge.

## Run locally
Python 3.10+:
```bash
pip install -r requirements.txt
python us_market_ai_v3.py --period 10y --max-symbols 1200
```

## GitHub Actions
The workflow installs VADER sentiment dependencies, runs after model/factor/workflow changes are pushed to `main`, and runs weekdays at 22:30 UTC (about 06:30 next day in China, UTC+8). GitHub may delay scheduled jobs. You can run manually from Actions → Daily US Market AI Scan → Run workflow. Download `daily-market-predictions` from the completed run's Artifacts section.

## Replace files in your repository
Upload `us_market_ai_v3.py` and `requirements.txt` to the repository root. Replace `.github/workflows/daily_scan.yml` with the included workflow. The workflow now runs V3. You may keep the old V2 file as an archive.
