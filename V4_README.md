# V4 Market Risk & Forecast Engine

V4 adds a separate daily research report alongside the existing V3.1 broad-stock opportunity scanner.

## Outputs
- \`market_forecasts_v4.csv\`: NDX proxy and SOXL 5/10/20-session forecasts.
- \`walkforward_validation_v4.csv\`: chronological holdout diagnostics (AUC, Brier score, directional accuracy, sample sizes).
- \`market_stress_proxies_v4.csv\` and \`market_stress_summary_v4.csv\`: VIX, HYG/LQD and NDX trend proxies.
- \`ai_company_fundamentals_v4.csv\`: latest available Yahoo Finance quarterly financial statement fields for selected AI/cloud companies.
- \`ai_news_v4.csv\`: recent GDELT headlines, source domain and article URL.
- \`market_dashboard_v4.md\`: human-readable report.
- \`run_summary_v4.json\`: run metadata and coverage.

## Method
- NQ is represented by the Nasdaq-100 cash index ticker \`^NDX\`, not Nasdaq-100 futures (\`NQ\`). This avoids pretending that cash-index data are futures quotes.
- NQ proxy and SOXL are modeled independently.
- Classifier ensemble: logistic regression + random forest. Expected return: histogram gradient boosting regression.
- Validation is chronological, with a forecast-horizon purge before the holdout segment.
- Current forecasts are fitted only on rows whose forward labels are already observable.
- Historical return quantiles are empirical forward-return distributions, not calibrated confidence intervals.
- Market stress score is a transparent rule-based proxy, not a probability of a crash.
- HYG/LQD is a credit-risk appetite proxy; it is not a substitute for CCC spreads or CDS.
- News is current-only and not inserted into historical training, avoiding a fake historical news backtest.
- Financial statement fields vary by issuer; missing values remain null.

## Not yet connected
- Options-chain GEX, dealer gamma, put/call walls, 0DTE positioning.
- Official FRED CCC spreads/CDS feed.
- Full historical constituents with delisted securities (survivorship-bias control).
- Calibrated probabilities and multi-period nested walk-forward validation.
- A complete whole-market NQ futures data feed.

## Run
\`\`\`bash
pip install -r requirements.txt
python market_risk_v4.py
\`\`\`

This is a research prototype, not investment advice. No predictive edge is assumed until the out-of-sample reports support it.
