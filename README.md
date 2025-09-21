# QuantTradSys

量化交易系統原型，涵蓋資料取得、策略回測與即時訊號模組。當前重點為加密資產 5 分鐘級距的一年期資料蒐集與快取。

## 目前里程碑
- ✅ CCXT 5 分鐘 / 一年期資料抓取元件（Functional Programming 實作）
- ✅ SQLite 持久層與增量更新流程（自動清理超過觀察窗口的舊資料）

## 資料抓取元件使用說明
1. 安裝依賴：`pip install -r requirements.txt`
2. 設定 Python 匯入路徑（範例 PowerShell）：`$env:PYTHONPATH = 'src'`
3. 以程式呼叫 `data_pipeline.ccxt_fetcher.fetch_yearly_ohlcv`。

```python
from pathlib import Path
from data_pipeline.ccxt_fetcher import fetch_yearly_ohlcv

btc_df = fetch_yearly_ohlcv(
    symbol="BTC/USDT",
    timeframe="5m",
    lookback_days=365,
    output_path=Path("data/BTCUSDT_5m.csv"),
    db_path=Path("storage/market_data.db"),
    exchange_config={
        "apiKey": "<optional>",
        "secret": "<optional>",
    },
)
print(btc_df.tail())
```

### 函式重點
- `fetch_yearly_ohlcv`：
  - 首次執行會抓取完整一年資料並寫入 SQLite；之後僅同步缺口。
  - 預設路徑 `storage/market_data.db`，包含 `symbol + timeframe + ts` 複合主鍵。
  - `prune_history=True` 時，會自動刪除觀察窗口前的舊資料。
- `save_dataframe`：若提供 `output_path`，會輸出最近一年視窗資料供回測使用。
- Logging 為英文，預設等級 INFO，可透過 `configure_logging(level=logging.DEBUG)` 觀察抓取進度與增量資訊。

## 後續規劃
- [ ] 新增資料完整性檢查（缺值 / 重複）
- [ ] 封裝資料快取排程（APS / Cron）
- [ ] 串接回測框架與策略倉儲流程
