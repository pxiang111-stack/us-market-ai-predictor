# DeepSeek量化委员会｜30只标的盘前/盘中报告

- 当前交易时段：**盘中 REGULAR**（美东）
- 最新可验证1分钟行情时间：**2026-11-27T00:00:00+00:00**
- UTC生成时间：2026-11-27T18:21:35.126911+00:00
- 行情源：Binance TradFi 1分钟实时行情；无法访问时使用Yahoo Finance 3mo日线备用模式（非实时）
- Binance受限时自动降级Yahoo日线参考；日线降级模式强制NO-GO，不视为实时行情
- 账户资金：10,000 USD；单笔风险预算：100 USD（1%）

## 一、通过标的（GO / GO SMALLER）

本次没有通过标的。

## 二、NO-GO淘汰汇总

| 代码 | 当前参考现价 | 一句话淘汰原因 |
|---|---:|---|
| SPY | 728.3200 | 行情非实时且data_status非OK，按规则不可用于GO/GO SMALLER；且价格在MA20/MA50下方，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| QQQ | 650.9900 | data_status=DAILY_ONLY且quote_verified_realtime=false，不满足实时报价门槛；中期均线空头排列，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| IWM | 273.7900 | 非实时报价且data_status非OK，无法验证入场；缩量反弹证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| SMH | 440.0600 | 报价非实时且data_status=DAILY_ONLY；均线深度空头排列，属破位后弱反弹，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| NVDA | 150.4900 | 非实时报价且data_status非OK；深度破位、缩量反弹，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| AAPL | 358.5400 | 虽结构偏多，但报价非实时且data_status非OK，无法验证实时入场，按规则NO-GO。；Python硬风控否决：币安无该标的合约行情。 |
| MSFT | 492.8500 | 非实时报价且data_status非OK；均线下方缩量，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| AMZN | 254.1200 | 非实时报价且data_status非OK，无法验证入场；量能不足，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| META | 628.9600 | 报价非实时且data_status=DAILY_ONLY；MA50下方反弹，无法验证实时入场，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| GOOGL | 274.9600 | 非实时报价且data_status非OK；MA50深度下方，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| TSLA | 400.0000 | 报价非实时且data_status=DAILY_ONLY，无法验证实时入场；消息驱动高估值，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| AMD | 424.5400 | 非实时报价且data_status非OK；深度破位且5日大幅下跌，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| AVGO | 312.3400 | 非实时报价且data_status非OK；5日大跌且均线下方，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| MU | 646.9500 | 非实时报价且data_status非OK；深度破位高波动，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| ARM | 171.5500 | 非实时报价且data_status非OK；SoftBank抛售叠加深度破位，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| TSM | 350.3600 | 非实时报价且data_status非OK；均线下方缩量，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| PLTR | 172.1700 | 报价非实时且data_status=DAILY_ONLY，无法验证实时入场；虽强势但证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| SMCI | 23.7600 | 非实时报价且data_status非OK；深度破位叠加供应链利空，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| NFLX | 60.9300 | 非实时报价且data_status非OK；MA50下方缩量反弹，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| CRM | 197.5800 | 非实时报价且data_status非OK；极度缩量反弹，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| ORCL | 106.5100 | 非实时报价且data_status非OK；深度破位叠加AI债务风险，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| APP | 399.2100 | 报价非实时且data_status=DAILY_ONLY；虽强势但严重超买且缩量，无法验证实时入场，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| UBER | 66.8050 | 非实时报价且data_status非OK；均线下方缩量，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| COIN | 107.0400 | 非实时报价且data_status非OK；深度破位，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| MSTR | 81.6200 | 非实时报价且data_status非OK；极端破位叠加杠杆与增发风险，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| HOOD | 67.9200 | 非实时报价且data_status非OK；深度破位，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| GLD | 334.4000 | 非实时报价且data_status非OK；均线下方且创多年最差11月，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| SLV | 39.0250 | 非实时报价且data_status非OK；深度破位，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| TLT | 81.2700 | 报价非实时且data_status=DAILY_ONLY，无法验证实时入场；利率政策风险高，证据不足。；Python硬风控否决：币安无该标的合约行情。 |
| XLE | 61.6700 | 非实时报价且data_status非OK；均线下方缩量，做多证据不足。；Python硬风控否决：币安无该标的合约行情。 |
