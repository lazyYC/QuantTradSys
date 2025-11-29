# QuantTradSys

star_xgb 為核心的加密量化策略專案，涵蓋資料擷取、Optuna 調參、報表輸出與即時排程。

## 目錄結構

`
QuantTradSys/
├── scripts/                     # 指令入口皆以 PYTHONPATH=src 執行
│   ├── backfill_ohlcv.py         # CCXT 拉取 / 回補資料（含 iso_ts 欄位）
│   ├── train.py                  # 策略訓練與 Optuna 調參 (通用)
│   ├── report.py                 # 績效報表輸出 (通用)
│   ├── run_scheduler.py          # 即時排程器 (通用)
│   └── ...                       # 其他工具腳本
├── src/
│   ├── data_pipeline/            # CCXT × SQLite I/O
│   ├── optimization/             # Optuna 搜尋 + nested split
│   ├── persistence/              # 參數 / 交易 / 績效 / 狀態儲存
│   ├── strategies/               # 策略實作 (如 star_xgb)
│   ├── reporting/                # 報表組件
│   └── ...
├── storage/                     # SQLite、報表輸出、Optuna DB
└── ...
```

## 安裝與環境

```powershell
python -m venv .venv
. .venv/Scripts/Activate
pip install -r requirements.txt
$env:PYTHONPATH = 'src'
```

## 資料回補

```powershell
python scripts/backfill_ohlcv.py BTC/USDT:USDT 5m 365 ^
    --db storage/market_data.db --exchange binanceusdm --prune
```

## 策略訓練 (Train)

使用 `scripts/train.py` 進行策略訓練與參數搜尋。

```powershell
python scripts/train.py ^
    --strategy star_xgb ^
    --study-name test ^
    --symbol BTC/USDT:USDT --timeframe 5m ^
    --lookback-days 360 --test-days 30 ^
    --n-trials 50 ^
    --n-seeds 5 ^ # 每個 trial 會用幾個 seed 去訓練
    --storage sqlite:///storage/optuna_studies.db
```

- `--strategy`: 策略演算法名稱 (如 `star_xgb`)。
- `--study-name`: 實驗名稱 (如 `test3`)，用於區分不同參數設定或實驗。

## 報表輸出 (Report)

使用 `scripts/report.py` 產生詳細回測報告。

```powershell
python scripts/report.py ^
    --strategy star_xgb --study test3 ^
    --dataset all ^
    --start 2024-01-01 --end 2024-12-31 ^
    --output reports/report.html
```

- `--dataset`: 選擇報告資料集 (`train`, `valid`, `test`, `all`)。
- `--start` / `--end`: 指定報告的時間範圍。

## 即時排程 (Scheduler)

使用 `scripts/run_scheduler.py` 啟動即時交易引擎。

```powershell
python scripts/run_scheduler.py ^
    --strategy star_xgb --study test ^
    --symbol BTC/USDT:USDT --timeframe 5m ^
    --lookback-days 60 ^
    --exchange binanceusdm
```

- 必須指定 `--strategy` 與 `--study` 以載入對應的訓練參數與模型。
- Scheduler 自動抓資料、產生訊號、並更新 runtime 狀態。

## 資料庫一覽

| DB | 用途 | 主要欄位 | 來源 |
| --- | --- | --- | --- |
| storage/market_data.db | OHLCV 快取 | ts, iso_ts, open~volume | backfill_ohlcv.py |
| storage/strategy_state.db | 策略參數 / 交易 / 績效 / runtime | strategy_params, strategy_trades, strategy_metrics, strategy_runtime | train.py, report.py, run_scheduler.py |
| storage/optuna_*.db | Optuna Study | trials, trial_params, trial_values | train.py |

## 開發守則

- 註解使用繁體中文；日誌使用英文，INFO 記錄重要事件、DEBUG 用於調試。
- 資料/訊號邏輯盡量採純函式以利測試與重構。
- 策略實作放入 src/strategies，並共用統一的接口。
- 中英文檔案請以 UTF-8 儲存，查看建議使用 scripts/cat_utf8.ps1 或設定 PowerShell OutputEncoding。

