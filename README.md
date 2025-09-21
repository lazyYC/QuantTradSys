# QuantTradSys

量化交易系統原型，涵蓋資料取得、策略回測與即時訊號模組。當前重點為加密資產 5 分鐘級距的一年期資料蒐集與快取。

## 目前里程碑
- ✅ CCXT 5 分鐘 / 一年期資料抓取元件（Functional Programming 實作）
- ✅ SQLite 持久層與增量更新流程（自動清理超過觀察窗口的舊資料）
- ✅ 訊號產生管線（函數式組合，可整合 rule-based / ML 評分器）
- ✅ 技術指標快速回測 + 即時訊號流程（基礎網格搜尋）

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

## 訊號產生管線使用說明
```python
import pandas as pd
from signals.generator import (
    FeatureStep,
    Scorer,
    generate_signal_frame,
)

# 簡單特徵步驟：新增移動平均
feature_steps: list[FeatureStep] = [
    lambda frame: frame.assign(ma_fast=frame["close"].rolling(window=5).mean()),
    lambda frame: frame.assign(ma_slow=frame["close"].rolling(window=20).mean()),
]

# 規則型評分器：黃金交叉買、死亡交叉賣

def crossover_scorer(row: pd.Series) -> dict | None:
    if pd.isna(row.ma_fast) or pd.isna(row.ma_slow):
        return None
    if row.ma_fast > row.ma_slow:
        return {"name": "ma_golden", "decision": "BUY", "confidence": 1.0}
    if row.ma_fast < row.ma_slow:
        return {"name": "ma_dead", "decision": "SELL", "confidence": 1.0}
    return None

scorers: list[Scorer] = [crossover_scorer]

df = pd.read_csv("data/BTCUSDT_5m.csv")
signal_df = generate_signal_frame(df, feature_steps, scorers)
print(signal_df[["timestamp", "signal"]].tail())
```

## 快速回測與即時訊號
```python
from pipelines.quick_backtest import quick_backtest_and_signal

result = quick_backtest_and_signal(
    symbol="BTC/USDT",
    timeframe="5m",
    lookback_days=30,
    output_path=None,
)
print(result['rankings'][0])
print('latest signal', result['signal'])
```

- `quick_backtest_and_signal`：
  - 透過 `pipelines.quick_backtest.DEFAULT_GRID` 的參數組合進行網格搜尋，找出年化報酬最佳的組合。
  - 以最佳參數呼叫 `generate_realtime_signal`，並透過 `notifier.dispatcher.dispatch_signal` 在日誌中宣告最新操作。
  - 可調整 `grid` 或 `lookback_days`，快速驗證不同區間的表現。

### 管線特色
- 特徵工程、評分器、決策解析器皆為函數，可依策略需求自由組合或替換。
- 預設 `weighted_vote_resolver` 採用加權信心投票，也可自行實作解析器以支援機器學習輸出（例如機率得分）。
- 產出 DataFrame 會附上 `evaluations` 欄位，保留每一列的評分來源與信心，方便除錯與回測。

## 後續規劃
- [ ] 新增資料完整性檢查（缺值 / 重複）
- [ ] 封裝資料快取排程（APS / Cron）
- [ ] 串接回測框架與策略倉儲流程


