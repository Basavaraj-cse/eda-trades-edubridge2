# Cleaning log

Every cleaning decision, how many rows it touched, and why.

| Issue | Rows affected | Decision | Reason |
|---|---|---|---|
| Duplicate rows | 35 | Dropped (keep first by trade_id) | They were exact copies; a trade must appear once or volume is double counted. |
| Symbol variants | 60 | Mapped to BTCUSDT | Same instrument written 4 ways; upper-case and remove / and -. |
| Side label variants | 119 | Lower-cased to buy / sell | BUY and buy are the same class. |
| Price stored as text ($ and ,) | 50 | Stripped $ and , then converted to float | Text prices cannot be sorted, averaged or compared with the book. |
| Mixed timestamp formats | 118 | Parsed to one datetime (UTC, naive) | DD/MM/YYYY read day-first; 10-digit Unix values read as seconds. |
| Rows in random order | 2000 | Sorted by timestamp then trade_id | Rolling features, returns and candles need time order. |
| Negative price | 6 | Sign flipped (abs) | |price| matched the best bid/ask to within 0.2 in every case, so only the sign was wrong. |
| Extreme price (x100) | 8 | Divided by 100 | After the fix every one is within 0.01% of the rolling median, so it was a scale error, not a real move. |
| Zero quantity | 6 | Set to NaN (not filled) | A trade cannot have zero size and the real size is unknown. Filling with a mean would invent volume. |
| Crossed order book | 10 | bid_price_1 and ask_price_1 set to NaN | A bid above the ask is impossible. We cannot tell which side is wrong, so spread, mid and imbalance are left empty for these rows. The trade itself is kept. |
| Missing side | 15 | Recovered 14 from the book, rest left NaN | In clean rows a sell prints at the best bid and a buy at the best ask. |
| Missing price | 25 | Recovered 25 from the book, rest left NaN | Same rule as above (sell = best bid, buy = best ask). |
| Missing ask_price_1 | 20 | Recovered 8 (buy trades only), rest left NaN | A buy prints at the best ask. For sells the ask is not determined by the price, so it stays NaN. |
| Missing quantity / bid_qty_1 | 50 | Left as NaN | No rule can recover a size. Imputing would bias volume and imbalance; rows are dropped only at modelling time. |

Rows in: 2035  ->  rows out: 2000 (unique trade_id: 2000)
