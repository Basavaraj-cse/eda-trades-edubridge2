# Tasks and process: Trading data EDA (synthetic BTCUSDT)

Dataset: `data/raw/trades_orderbook_STUDENT_dirty.csv` (2,035 rows, deliberately messy).
Each row is one trade plus the top 3 bid and ask levels of the order book at that moment.
All data is synthetic. Educational material, not financial advice.

## What each student must do

1. Load the file and report **all** data quality issues found.
2. Clean it (Python/pandas or Excel) and **explain every decision** (drop vs fill vs fix).
3. Sort by time and remove duplicates by `trade_id`.
4. Add columns: `spread = ask_1 - bid_1`, `mid_price = (ask_1 + bid_1) / 2`,
   `order_imbalance = (bid_qty_1 - ask_qty_1) / (bid_qty_1 + ask_qty_1)`.
5. Resample to 1-minute OHLCV candles.

## Process (from `EDA_Steps_Trading_Data.pdf`)

| # | Step | What to do | Status | Where |
|---|------|-----------|--------|-------|
| 1 | Load and inspect | `read_csv`, `head()`, `shape`, `info()`, dtypes | Done | `src/eda_pipeline.py` |
| 2 | Summarize | `describe()`, `value_counts()` on side and symbol | Done | same |
| 3 | Find the problems | Missing, duplicates, mixed timestamps, negative/zero, crossed books | Done | `reports/data_quality_issues.csv`, `reports/figures/00_data_quality_issues.png` |
| 4 | Clean | Dedupe by `trade_id`, standardize side/symbol, price text to number, one timestamp format, fix or drop bad rows | Done | `data/processed/trades_clean.csv`, `reports/cleaning_log.md` |
| 5 | Outliers | Rolling median (plus IQR check) for extreme prices | Done | `reports/figures/01_price_before_after.png` |
| 6 | Features | spread, mid_price, order_imbalance, returns, rolling volatility, MA, RSI | Done | `data/processed/trades_features.csv` |
| 7 | Resample | 1-minute OHLCV candles sorted by time | Done | `data/processed/ohlcv_1min.csv` |
| 8 | Visualize | Price line, volume histogram, spread distribution, buy vs sell, correlation heatmap | Done | `reports/figures/` |
| 9 | Prepare for a model | Target (price up in 5 min), time split 70/30, scaling, save files | Done | `data/processed/model_*.csv` |

## Rules for students

- Never shuffle time-series data. Split by time only.
- Avoid look-ahead bias: no feature may use future data. Only the target looks forward.
- Document every cleaning decision (see `reports/cleaning_log.md`).

## Demo path (3Commas)

Cleaned data -> features -> simple rule (for example "buy when order imbalance > 0.3")
-> backtest -> buy/sell signals by webhook to 3Commas, **paper trading first**.
This repo stops at the simple rule check (`reports/rule_check_imbalance_gt_0_3.csv`).
A real backtest needs fees, slippage and much more data.
