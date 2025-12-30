# QuantTradSys

star_xgb 為核心的加密量化策略專案，涵蓋資料擷取、Optuna 調參、報表輸出與即時排程。

## 目錄結構

```
QuantTradSys/
├── scripts/                  # CLI Entry Points
│   ├── backfill_ohlcv.py     # 歷史資料回補 (CCXT -> Postgres)
│   ├── train.py              # 策略訓練與最佳化 (Training Engine)
│   ├── report.py             # 績效報告產生 (Reporting Engine)
│   └── run_scheduler.py      # 即時交易排程 (Realtime Engine)
├── src/
│   ├── config/               # 環境變數與路徑配置
│   ├── database/             # 資料庫核心 (PostgreSQL)
│   │   ├── connection.py     # 連線與 Session 管理
│   │   └── schema.py         # SQLAlchemy ORM 定義 (OHLCV, Trade, Param...)
│   ├── persistence/          # 資料存取層 (Repository Pattern)
│   │   ├── market_store.py   # OHLCV 資料存取
│   │   ├── trade_store.py    # 交易紀錄存取
│   │   ├── param_store.py    # 策略參數存取
│   │   └── runtime_store.py  # 即時狀態存取
│   ├── data_pipeline/        # 資料擷取與清理
│   │   ├── ccxt_fetcher.py   # CCXT 介面 (Fetch & Sync)
│   │   └── reader.py         # 資料讀取 (從 MarketStore)
│   ├── strategies/           # 策略實作目錄
│   │   ├── base.py           # 策略基底類別
│   │   └── star_xgb/         # 範例策略
│   ├── training/             # 訓練引擎 (Workflow)
│   ├── reporting/            # 報告引擎 (Workflow)
│   ├── engine/               # 即時引擎 (Realtime Workflow)
│   └── utils/                # 通用工具 (Logging, Formatting...)
├── storage/                  # 包含 logs 與 lock files (DB 已遷移至雲端)
└── README.md
```

## 安裝與環境

```powershell
python -m venv .venv
. .venv/Scripts/Activate
pip install -r requirements.txt
python -m venv .venv
. .venv/Scripts/Activate
pip install -r requirements.txt
pip install -e .  # 安裝為開發模式，自動處理路徑
```

## 資料回補

```powershell
python scripts/backfill_ohlcv.py BTC/USDT:USDT 5m 365 --exchange binanceusdm --prune
```

## 策略訓練 (Train)

使用 `scripts/train.py` 進行策略訓練與參數搜尋。

```powershell
python scripts/train.py --strategy star_xgb --study-name test --symbol BTC/USDT:USDT --timeframe 5m --lookback-days 360 --test-days 30 --n-trials 50 --n-seeds 5 # 每個 trial 會用幾個 seed 去訓練
```

- `--strategy`: 策略演算法名稱 (如 `star_xgb`)。
- `--study-name`: 實驗名稱 (如 `test3`)，用於區分不同參數設定或實驗。

## 報表輸出 (Report)

使用 `scripts/report.py` 產生詳細回測報告。

```powershell
python scripts/report.py --strategy star_xgb --study test3 --dataset all --start 2024-01-01 --end 2024-12-31 --output reports/report.html
```

- `--dataset`: 選擇報告資料集 (`train`, `valid`, `test`, `all`)。
- `--start` / `--end`: 指定報告的時間範圍。

## 即時排程 (Scheduler)

使用 `scripts/run_scheduler.py` 啟動即時交易引擎。

```powershell
python scripts/run_scheduler.py --strategy star_xgb --study test --symbol BTC/USDT:USDT --timeframe 5m --lookback-days 60 --exchange binanceusdm
```

- 指定 `--strategy` 與 `--study` 以載入對應的訓練參數與模型。
- Scheduler 自動抓資料、產生訊號、並更新 runtime 狀態。

