# EDA on trading data (synthetic BTCUSDT)

Exploratory data analysis, cleaning and feature engineering on a deliberately messy
trades + order book file. Built for the internship EDA program.
The task list and process are in [`README_tasks.md`](README_tasks.md).

> Educational material, not financial advice. The data is synthetic.

## Repo layout

```
data/raw/        trades_orderbook_STUDENT_dirty.csv   original file, never edited
data/processed/  trades_clean.csv        cleaned trades (2,000 rows, with fix_flags column)
                 trades_features.csv     + spread, mid_price, order_imbalance, returns, volatility
                 ohlcv_1min.csv          1-minute candles with MA and RSI
                 model_dataset.csv       features + 5-minute target + train/test split column
                 model_train_scaled.csv / model_test_scaled.csv
notebooks/       eda_trades.ipynb        same code as the script, as a notebook
src/             eda_pipeline.py         full pipeline, steps 1 to 9
reports/         cleaning_log.md         every cleaning decision and why
                 data_quality_issues.csv, rule_check_imbalance_gt_0_3.csv
                 figures/                7 charts
```

## How to run

```bash
pip install -r requirements.txt
python src/eda_pipeline.py        # rebuilds everything in data/processed and reports/
```

## What was found (2,035 rows in, 2,000 rows out)

| Issue | Count | Decision |
|---|---|---|
| Duplicate rows (same `trade_id`) | 35 | Dropped |
| Mixed timestamps (DD/MM/YYYY, Unix) | 78 + 40 | Parsed to one datetime |
| Side labels (BUY, Sell) / symbol variants | 119 / 60 | Standardized to `buy`,`sell` / `BTCUSDT` |
| Price stored as text (`$62,184.2`) | 50 | Converted to float |
| Extreme prices (about x100) | 8 | Divided by 100 (verified against rolling median) |
| Negative prices | 6 | Sign flipped (matched the book) |
| Zero quantity | 6 | Set to NaN |
| Crossed book (bid above ask) | 10 | Top of book set to NaN, trade kept |
| Missing side / price / ask_price_1 | 15 / 25 / 20 | Recovered from the book where provable (14 / 25 / 8) |
| Missing quantity / bid_qty_1 | 20 / 30 | Left as NaN, never imputed |
| Rows out of time order | about half | Sorted by time |

Notes worth knowing:

- The 10-digit Unix timestamps are **seconds**, not milliseconds.
- `DD/MM/YYYY` is day-first (all of them are 1 October 2026).
- In clean rows a sell prints at the best bid and a buy at the best ask. That rule is what
  allowed missing prices and sides to be recovered. Every such fix is tagged in `fix_flags`.
- No row was dropped except duplicates. Rows with unrecoverable NaN are dropped only when
  building the model dataset (1,548 of 2,000 rows are usable, because the last 5 minutes have no target).

## Results of the simple model step

- Data covers about 27 minutes (09:00 to 09:27 on 2026-10-01), so the 1-minute candle file has 28 rows.
- Target: price higher 5 minutes after each trade. Split by time: first 70% train, last 30% test.
- Rule "buy when order imbalance > 0.3": hit rate 74.5% on the test set versus 73.5% for all rows,
  and correlation of imbalance with the 5-minute return is about -0.02.
  **The signal adds nothing here.** The high hit rate only reflects the upward drift in this short synthetic sample.

## Limits

- Short, synthetic sample: conclusions do not transfer to real markets.
- The rule check is not a backtest (no fees, slippage or position sizing).
- Timestamps are treated as naive UTC because the file gives no timezone.
