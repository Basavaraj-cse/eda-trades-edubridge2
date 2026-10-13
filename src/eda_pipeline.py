# %% [markdown]
# # EDA on synthetic BTCUSDT trades + order book (dirty student dataset)
#
# Follows `EDA_Steps_Trading_Data.pdf` step by step (1 to 9).
# Run from the repo root: `python src/eda_pipeline.py`
# Educational material, not financial advice. The data is synthetic.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside a notebook
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

RAW = ROOT / "data/raw/trades_orderbook_STUDENT_dirty.csv"
PROC = ROOT / "data/processed"
FIG = ROOT / "reports/figures"
REP = ROOT / "reports"
for p in (PROC, FIG, REP):
    p.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.show()


# %% [markdown]
# ## Step 1. Load and inspect

# %%
raw = pd.read_csv(RAW)
print("shape:", raw.shape)
print(raw.head())
raw.info()

# %% [markdown]
# ## Step 2. Summarize

# %%
print(raw.describe().T)
print(raw["side"].value_counts(dropna=False))
print(raw["symbol"].value_counts(dropna=False))

# %% [markdown]
# ## Step 3. Find the problems
#
# Helper functions first. They are reused when cleaning, so the detection and
# the fix always use the same rules.

# %%
TS_STD = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}"
TS_DMY = r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}"
TS_UNIX = r"\d{10}|\d{13}"


def ts_format(s):
    s = s.astype(str).str.strip()
    return pd.Series(
        np.select(
            [s.str.fullmatch(TS_STD), s.str.fullmatch(TS_DMY), s.str.fullmatch(TS_UNIX)],
            ["standard", "dd/mm/yyyy", "unix"],
            default="unknown",
        ),
        index=s.index,
    )


def parse_timestamps(s):
    """Parse the 3 timestamp formats into one naive UTC datetime64[ns]."""
    s = s.astype(str).str.strip()
    fmt = ts_format(s)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    m = fmt == "standard"
    out[m] = pd.to_datetime(s[m], format="%Y-%m-%d %H:%M:%S.%f").astype("datetime64[ns]")
    m = fmt == "dd/mm/yyyy"
    out[m] = pd.to_datetime(s[m], format="%d/%m/%Y %H:%M:%S").astype("datetime64[ns]")
    m = fmt == "unix"
    # 10 digits = seconds, 13 digits = milliseconds
    secs = s[m].str.len() == 10
    ints = s[m].astype("int64")
    out[m] = pd.to_datetime(np.where(secs, ints * 1000, ints), unit="ms").astype("datetime64[ns]")
    return out


def to_number(s):
    """'$62,184.2' -> 62184.2 ; unparseable -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def price_outliers(price, window=11, ratio=3.0):
    """Rolling-median test: price more than 3x above / below the local median."""
    med = price.where(price > 0).rolling(window, center=True, min_periods=3).median()
    r = price / med
    return (r > ratio) | (r < 1 / ratio), med


# %%
# Value-level defects are counted on the de-duplicated trades so that the
# 35 duplicate rows do not inflate the counts.
dedup = raw.drop_duplicates(subset="trade_id", keep="first").copy()
dedup["ts"] = parse_timestamps(dedup["timestamp"])
dedup = dedup.sort_values(["ts", "trade_id"]).reset_index(drop=True)
dedup["price_num"] = to_number(dedup["price"])
out_mask, _ = price_outliers(dedup["price_num"])

issues = pd.DataFrame(
    [
        ("Exact duplicate rows", int(raw.duplicated().sum()), "rows"),
        ("Duplicate trade_id", int(raw["trade_id"].duplicated().sum()), "rows"),
        ("Missing side", int(dedup["side"].isna().sum()), "trades"),
        ("Missing price", int(dedup["price_num"].isna().sum()), "trades"),
        ("Missing quantity", int(dedup["quantity"].isna().sum()), "trades"),
        ("Missing bid_qty_1", int(dedup["bid_qty_1"].isna().sum()), "trades"),
        ("Missing ask_price_1", int(dedup["ask_price_1"].isna().sum()), "trades"),
        ("Timestamp format: DD/MM/YYYY", int((ts_format(dedup["timestamp"]) == "dd/mm/yyyy").sum()), "trades"),
        ("Timestamp format: Unix epoch", int((ts_format(dedup["timestamp"]) == "unix").sum()), "trades"),
        ("Side label not lower-case (BUY, Sell)", int((dedup["side"].dropna() != dedup["side"].dropna().str.lower()).sum()), "trades"),
        ("Symbol not 'BTCUSDT' (btcusdt, BTC/USDT, BTC-USDT)", int((dedup["symbol"] != "BTCUSDT").sum()), "trades"),
        ("Price stored as text with $ or ,", int(dedup["price"].astype(str).str.contains(r"[$,]").sum()), "trades"),
        ("Negative price", int((dedup["price_num"] < 0).sum()), "trades"),
        ("Extreme price (rolling median, about x100)", int((out_mask & (dedup["price_num"] > 0)).sum()), "trades"),
        ("Zero quantity", int((dedup["quantity"] == 0).sum()), "trades"),
        ("Crossed book (bid_1 > ask_1)", int((dedup["bid_price_1"] > dedup["ask_price_1"]).sum()), "trades"),
        ("Rows not in time order", int((parse_timestamps(raw["timestamp"]).diff().dt.total_seconds() < 0).sum()), "rows"),
    ],
    columns=["issue", "count", "unit"],
)
print(issues.to_string(index=False))
issues.to_csv(REP / "data_quality_issues.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=issues.sort_values("count"), y="issue", x="count", color="#c0504d", ax=ax)
ax.set_title("Data quality issues found")
ax.set_ylabel("")
save_fig(fig, "00_data_quality_issues.png")

# %% [markdown]
# Notes from the inspection:
# * The Unix timestamps all have **10 digits**, so they are **seconds**, not milliseconds.
#   Reading them as milliseconds would put those trades in January 1970.
# * The `DD/MM/YYYY` rows are day-first: the first field is always `01` and the second `10`,
#   which can only be 1 October 2026 (month 01 does not exist in the other formats).
# * In clean rows `price == bid_price_1` for sells and `price == ask_price_1` for buys.
#   That lets us recover some missing prices and sides from the book.

# %% [markdown]
# ## Steps 4 and 5. Clean and handle outliers
#
# Every fix is flagged in the `fix_flags` column and counted in the cleaning log.

# %%
df = raw.copy()
log = []  # (issue, rows affected, decision, reason)

# 4a. duplicates ---------------------------------------------------------
n = len(df)
df = df.drop_duplicates(subset="trade_id", keep="first")
log.append(("Duplicate rows", n - len(df), "Dropped (keep first by trade_id)",
            "They were exact copies; a trade must appear once or volume is double counted."))

# 4b. standardize categorical columns --------------------------------------
df["symbol"] = df["symbol"].str.upper().str.replace(r"[/\-]", "", regex=True)
log.append(("Symbol variants", int((raw.drop_duplicates('trade_id')["symbol"] != "BTCUSDT").sum()), "Mapped to BTCUSDT",
            "Same instrument written 4 ways; upper-case and remove / and -."))
df["side"] = df["side"].str.strip().str.lower()
log.append(("Side label variants", int((raw.drop_duplicates('trade_id')["side"].dropna() != raw.drop_duplicates('trade_id')["side"].dropna().str.lower()).sum()),
            "Lower-cased to buy / sell", "BUY and buy are the same class."))

# 4c. price text -> number ---------------------------------------------------
n_text = int(df["price"].astype(str).str.contains(r"[$,]").sum())
df["price"] = to_number(df["price"])
log.append(("Price stored as text ($ and ,)", n_text, "Stripped $ and , then converted to float",
            "Text prices cannot be sorted, averaged or compared with the book."))

# 4d. timestamps -------------------------------------------------------------
fmt = ts_format(df["timestamp"])
df["timestamp"] = parse_timestamps(df["timestamp"])
log.append(("Mixed timestamp formats", int((fmt != "standard").sum()),
            "Parsed to one datetime (UTC, naive)",
            "DD/MM/YYYY read day-first; 10-digit Unix values read as seconds."))
df = df.sort_values(["timestamp", "trade_id"]).reset_index(drop=True)
log.append(("Rows in random order", len(df), "Sorted by timestamp then trade_id",
            "Rolling features, returns and candles need time order."))

flags = pd.DataFrame(False, index=df.index, columns=[
    "price_sign_fixed", "price_x100_fixed", "qty_invalid_to_nan", "book_crossed_to_nan",
    "side_from_book", "price_from_book", "ask1_from_price"])

# 5a. negative prices -> sign flipped ----------------------------------------
neg = df["price"] < 0
df.loc[neg, "price"] = df.loc[neg, "price"].abs()
flags.loc[neg, "price_sign_fixed"] = True
log.append(("Negative price", int(neg.sum()), "Sign flipped (abs)",
            "|price| matched the best bid/ask to within 0.2 in every case, so only the sign was wrong."))

# 5b. extreme prices (x100) -> divided by 100 ----------------------------------
out, med = price_outliers(df["price"])
out = out & (df["price"] > 0)
df.loc[out, "price"] = df.loc[out, "price"] / 100
flags.loc[out, "price_x100_fixed"] = True
dev = ((df.loc[out, "price"] - med[out]).abs() / med[out]).max() if out.any() else 0
log.append(("Extreme price (x100)", int(out.sum()), "Divided by 100",
            f"After the fix every one is within {dev:.2%} of the rolling median, so it was a scale error, not a real move."))

# IQR double check on what is left
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr_out = ((df["price"] < q1 - 3 * (q3 - q1)) | (df["price"] > q3 + 3 * (q3 - q1))).sum()
print("IQR (3x) outliers left after cleaning:", int(iqr_out))

# 5c. zero quantity -> NaN ------------------------------------------------------
zq = df["quantity"] <= 0
df.loc[zq, "quantity"] = np.nan
flags.loc[zq, "qty_invalid_to_nan"] = True
log.append(("Zero quantity", int(zq.sum()), "Set to NaN (not filled)",
            "A trade cannot have zero size and the real size is unknown. Filling with a mean would invent volume."))

# 4e. crossed book -> top of book set to NaN --------------------------------------
cross = df["bid_price_1"] >= df["ask_price_1"]
df.loc[cross, ["bid_price_1", "ask_price_1"]] = np.nan
flags.loc[cross, "book_crossed_to_nan"] = True
log.append(("Crossed order book", int(cross.sum()), "bid_price_1 and ask_price_1 set to NaN",
            "A bid above the ask is impossible. We cannot tell which side is wrong, so spread, mid and imbalance are left empty for these rows. The trade itself is kept."))

# 4f. recover what the book can prove -----------------------------------------------
# side from price (price == bid_1 -> sell, price == ask_1 -> buy)
m = df["side"].isna() & df["price"].notna()
as_sell = m & np.isclose(df["price"], df["bid_price_1"])
as_buy = m & np.isclose(df["price"], df["ask_price_1"])
df.loc[as_sell, "side"] = "sell"
df.loc[as_buy, "side"] = "buy"
flags.loc[as_sell | as_buy, "side_from_book"] = True
log.append(("Missing side", int(raw.drop_duplicates('trade_id')["side"].isna().sum()), f"Recovered {int((as_sell | as_buy).sum())} from the book, rest left NaN",
            "In clean rows a sell prints at the best bid and a buy at the best ask."))

# price from side + book
m = df["price"].isna()
p_sell = m & (df["side"] == "sell") & df["bid_price_1"].notna()
p_buy = m & (df["side"] == "buy") & df["ask_price_1"].notna()
df.loc[p_sell, "price"] = df.loc[p_sell, "bid_price_1"]
df.loc[p_buy, "price"] = df.loc[p_buy, "ask_price_1"]
flags.loc[p_sell | p_buy, "price_from_book"] = True
log.append(("Missing price", int(m.sum()), f"Recovered {int((p_sell | p_buy).sum())} from the book, rest left NaN",
            "Same rule as above (sell = best bid, buy = best ask)."))

# ask_price_1 from price for buys
m = df["ask_price_1"].isna() & (df["side"] == "buy") & df["price"].notna() & ~cross
df.loc[m, "ask_price_1"] = df.loc[m, "price"]
flags.loc[m, "ask1_from_price"] = True
log.append(("Missing ask_price_1", int(raw.drop_duplicates('trade_id')["ask_price_1"].isna().sum()), f"Recovered {int(m.sum())} (buy trades only), rest left NaN",
            "A buy prints at the best ask. For sells the ask is not determined by the price, so it stays NaN."))

log.append(("Missing quantity / bid_qty_1", int(raw.drop_duplicates("trade_id")["quantity"].isna().sum() + raw.drop_duplicates("trade_id")["bid_qty_1"].isna().sum()), "Left as NaN",
            "No rule can recover a size. Imputing would bias volume and imbalance; rows are dropped only at modelling time."))

df["fix_flags"] = flags.apply(lambda r: ";".join(r.index[r]), axis=1)
clean = df.copy()
clean.to_csv(PROC / "trades_clean.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")

print("\nRows:", len(clean), "| unique trade_id:", clean["trade_id"].nunique())
print("Sorted by time:", clean["timestamp"].is_monotonic_increasing)
print("Missing after cleaning:\n", clean.isna().sum()[lambda s: s > 0])

# %% [markdown]
# ### Before vs after (price)

# %%
before = dedup.set_index("ts")["price_num"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(before.index, before.values, lw=0.7, color="#c0504d")
axes[0].set_title("Price BEFORE cleaning (spikes and negatives)")
axes[1].plot(clean["timestamp"], clean["price"], lw=0.7, color="#1f77b4")
axes[1].set_title("Price AFTER cleaning")
for a in axes:
    a.tick_params(axis="x", rotation=30)
save_fig(fig, "01_price_before_after.png")

# %% [markdown]
# ## Step 6. Engineer features
# Every feature uses only the current and past rows (no look-ahead).

# %%
feats = clean.copy()
feats["spread"] = feats["ask_price_1"] - feats["bid_price_1"]
feats["mid_price"] = (feats["ask_price_1"] + feats["bid_price_1"]) / 2
feats["order_imbalance"] = (feats["bid_qty_1"] - feats["ask_qty_1"]) / (feats["bid_qty_1"] + feats["ask_qty_1"])
feats["log_return"] = np.log(feats["price"]).diff()
feats["volatility_20"] = feats["log_return"].rolling(20).std()  # past 20 trades only
feats["ma_20"] = feats["price"].rolling(20).mean()
feats.to_csv(PROC / "trades_features.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
print(feats[["spread", "mid_price", "order_imbalance", "log_return", "volatility_20"]].describe().T)

# %% [markdown]
# ## Step 7. Resample to 1-minute OHLCV candles

# %%
t = feats.set_index("timestamp").sort_index()
candles = t["price"].resample("1min").ohlc()
candles["volume"] = t["quantity"].resample("1min").sum(min_count=1)
candles["n_trades"] = t["price"].resample("1min").count()
candles["buy_volume"] = t.loc[t["side"] == "buy", "quantity"].resample("1min").sum()
candles["sell_volume"] = t.loc[t["side"] == "sell", "quantity"].resample("1min").sum()
candles["avg_spread"] = t["spread"].resample("1min").mean()
candles["avg_imbalance"] = t["order_imbalance"].resample("1min").mean()

# candle level indicators (past data only)
candles["log_return"] = np.log(candles["close"]).diff()
candles["volatility_5"] = candles["log_return"].rolling(5).std()
candles["ma_5"] = candles["close"].rolling(5).mean()
delta = candles["close"].diff()
gain = delta.clip(lower=0).ewm(alpha=1 / 5, min_periods=5, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1 / 5, min_periods=5, adjust=False).mean()
candles["rsi_5"] = 100 - 100 / (1 + gain / loss)  # short period: the sample is only ~30 candles

candles.to_csv(PROC / "ohlcv_1min.csv", date_format="%Y-%m-%d %H:%M:%S")
print(candles[["open", "high", "low", "close", "volume", "n_trades"]].head(), "\n...")
print("candles:", len(candles), "| empty candles:", int((candles["n_trades"] == 0).sum()))
print("time range:", candles.index.min(), "->", candles.index.max())

# %% [markdown]
# ## Step 8. Visualize

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(clean["timestamp"], clean["price"], lw=0.5, alpha=0.5, label="trade price")
ax.plot(candles.index, candles["close"], lw=2, label="1-min close")
ax.set_title("BTCUSDT price (synthetic)")
ax.legend()
save_fig(fig, "02_price_line.png")

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(clean["quantity"].dropna(), bins=40, ax=ax)
ax.set_title("Trade quantity distribution")
save_fig(fig, "03_volume_hist.png")

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(feats["spread"].dropna(), bins=20, ax=ax)
ax.set_title("Spread distribution (ask_1 - bid_1)")
save_fig(fig, "04_spread_dist.png")

fig, ax = plt.subplots(figsize=(4.5, 4))
sns.countplot(x=clean["side"].dropna(), order=["buy", "sell"], ax=ax)
ax.set_title("Buy vs sell trades")
save_fig(fig, "05_buy_vs_sell.png")

corr_cols = ["price", "quantity", "spread", "mid_price", "order_imbalance", "log_return", "volatility_20", "bid_qty_1", "ask_qty_1"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(feats[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation of features")
save_fig(fig, "06_correlation_heatmap.png")

# %% [markdown]
# ## Step 9. Prepare for a model
# Target: does the trade price go **up** 5 minutes after each trade?
# The target is the only column that looks into the future, and it is never used as a feature.
# The split is by time (first 70% train, last 30% test), never shuffled.

# %%
valid = feats["price"].notna()
future_lookup = feats.loc[valid, ["timestamp", "price"]].rename(columns={"price": "future_price"})
query = pd.DataFrame({"t_future": feats["timestamp"] + pd.Timedelta(minutes=5)})
query["row"] = query.index
merged = pd.merge_asof(query.sort_values("t_future"), future_lookup.sort_values("timestamp"),
                       left_on="t_future", right_on="timestamp", direction="backward")
feats["future_price"] = merged.set_index("row")["future_price"].reindex(feats.index)
feats.loc[query["t_future"] > feats["timestamp"].max(), "future_price"] = np.nan  # no data 5 min ahead
feats["future_return_5m"] = np.log(feats["future_price"] / feats["price"])
feats["target_up_5m"] = (feats["future_price"] > feats["price"]).astype("float").where(feats["future_price"].notna())
feats["side_is_buy"] = (feats["side"] == "buy").astype("float").where(feats["side"].notna())

feature_cols = ["spread", "order_imbalance", "log_return", "volatility_20", "quantity", "side_is_buy"]
model_df = feats.dropna(subset=feature_cols + ["target_up_5m"]).reset_index(drop=True)
split = int(len(model_df) * 0.7)
model_df["split"] = np.where(model_df.index < split, "train", "test")
train, test = model_df[model_df["split"] == "train"], model_df[model_df["split"] == "test"]
assert train["timestamp"].max() <= test["timestamp"].min(), "split must respect time order"

print(f"rows usable for modelling: {len(model_df)} of {len(feats)}")
print(f"train: {len(train)}  ({train['timestamp'].min()} -> {train['timestamp'].max()})")
print(f"test : {len(test)}  ({test['timestamp'].min()} -> {test['timestamp'].max()})")
print("share of 'up' in train / test:", round(train["target_up_5m"].mean(), 3), round(test["target_up_5m"].mean(), 3))

# scale with train statistics only (no leakage from the test period)
mu, sd = train[feature_cols].mean(), train[feature_cols].std()
scaled = model_df.copy()
scaled[feature_cols] = (model_df[feature_cols] - mu) / sd
keep = ["trade_id", "timestamp"] + feature_cols + ["future_return_5m", "target_up_5m", "split"]
model_df[keep].to_csv(PROC / "model_dataset.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
scaled.loc[scaled["split"] == "train", keep].to_csv(PROC / "model_train_scaled.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
scaled.loc[scaled["split"] == "test", keep].to_csv(PROC / "model_test_scaled.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")

# %% [markdown]
# ### Simple rule check: "buy when order imbalance > 0.3"
# Not a full backtest (no fees, no slippage, no position sizing). It only asks whether the signal
# had any link with the next 5 minutes on the held-out test period.

# %%
sig = test["order_imbalance"] > 0.3
rule = pd.DataFrame({
    "set": ["test"],
    "signals": [int(sig.sum())],
    "share_of_rows": [round(sig.mean(), 3)],
    "hit_rate_signal": [round(test.loc[sig, "target_up_5m"].mean(), 3)],
    "hit_rate_all_rows": [round(test["target_up_5m"].mean(), 3)],
    "avg_5m_return_signal": [test.loc[sig, "future_return_5m"].mean()],
    "avg_5m_return_all": [test["future_return_5m"].mean()],
})
print(rule.T)
print("corr(order_imbalance, 5m future return) on test:",
      round(test["order_imbalance"].corr(test["future_return_5m"]), 3))
rule.to_csv(REP / "rule_check_imbalance_gt_0_3.csv", index=False)

# %% [markdown]
# ## Write the cleaning log

# %%
cl = pd.DataFrame(log, columns=["issue", "rows_affected", "decision", "reason"])
cl.to_csv(REP / "cleaning_log.csv", index=False)
with open(REP / "cleaning_log.md", "w") as f:
    f.write("# Cleaning log\n\nEvery cleaning decision, how many rows it touched, and why.\n\n")
    f.write("| Issue | Rows affected | Decision | Reason |\n|---|---|---|---|\n")
    for r in log:
        f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n")
    f.write(f"\nRows in: {len(raw)}  ->  rows out: {len(clean)} (unique trade_id: {clean['trade_id'].nunique()})\n")
print(cl.to_string(index=False))
