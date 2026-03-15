"""
Pipeline configuration — all magic numbers live here.
"""
from pathlib import Path
from datetime import datetime, timezone

# ── Directory layout ──────────────────────────────────────────────────────────
RAW_DIR  = Path(__file__).resolve().parent   # data_center/raw_data/
DATA_DIR = RAW_DIR.parent                    # data_center/
CKPT_DIR = RAW_DIR / "checkpoints"
PARQ_DIR = DATA_DIR / "parquet"
DB_PATH  = DATA_DIR / "market_data.db"

# ── OKX REST API ──────────────────────────────────────────────────────────────
OKX_BASE         = "https://www.okx.com"
# history-candles: paginated deep history (up to 100 bars/request)
HISTORY_ENDPOINT = "/api/v5/market/history-candles"
# candles: recent data, up to 1440 bars/request (last ~5 days for 5m)
RECENT_ENDPOINT  = "/api/v5/market/candles"

BAR      = "5m"
BAR_MS   = 5 * 60 * 1000        # 300 000 ms per candle
LIMIT    = 100                   # OKX history-candles hard cap

# ── Rate-limit & retry ────────────────────────────────────────────────────────
# OKX public market data: 20 req / 2 s  →  safe ceiling ≈ 8 req/s
REQ_INTERVAL = 0.13              # seconds between consecutive requests
MAX_RETRIES  = 6
BACKOFF_BASE = 2.0               # exponential backoff multiplier (2^attempt s)

# ── Time range ────────────────────────────────────────────────────────────────
START_TS = int(datetime(2020,  1,  1,  0,  0,  0, tzinfo=timezone.utc).timestamp() * 1000)
END_TS   = int(datetime(2026,  3, 14, 23, 55,  0, tzinfo=timezone.utc).timestamp() * 1000)

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOLS = ["BTC-USDT", "ETH-USDT"]

# ── Cleaning ──────────────────────────────────────────────────────────────────
SPIKE_WINDOW = 50      # rolling window size for spike detection
SPIKE_SIGMA  = 5.0     # how many σ above rolling mean counts as a spike
