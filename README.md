# QuantTradSys

star_xgb 為核心的加密量化策略專案，涵蓋資料擷取、Optuna 調參、報表輸出與即時排程。

## 目錄結構

`
QuantTradSys/
├── scripts/                     # 指令入口皆以 PYTHONPATH=src 執行
│   ├── backfill_ohlcv.py         # CCXT 拉取 / 回補資料（含 iso_ts 欄位）
│   ├── pipeline_run_star_xgb.py  # star_xgb Optuna 管線
│   ├── render_star_xgb_report.py # Plotly 報表輸出
│   └── ...                       # 其他工具腳本
├── src/
│   ├── data_pipeline/            # CCXT × SQLite I/O
│   ├── optimization/             # Optuna 搜尋 + nested split
│   ├── persistence/              # 參數 / 交易 / 績效 / 狀態儲存
│   ├── strategies/star_xgb/      # 動能 K 線 + XGBoost 策略
│   ├── reporting/                # 報表組件
│   └── run_star_xgb_scheduler.py # 即時排程器
├── storage/                     # SQLite、報表輸出、Optuna DB
└── ...
`

## 安裝與環境

`powershell
python -m venv .venv
. .venv/Scripts/Activate
pip install -r requirements.txt
 = 'src'
`

## 資料回補

`powershell
python scripts/backfill_ohlcv.py BTC/USDT:USDT 5m 365 ^
    --db storage/market_data.db --exchange binanceusdm --prune
`

## Optuna 調參（star_xgb）

`powershell
python scripts/pipeline_run_star_xgb.py ^
    --symbol BTC/USDT:USDT --timeframe 5m ^
    --lookback-days 360 --test-days 30 ^
    --exchange binanceusdm ^
    --n-trials 80 ^
    --study-name star_xgb_prod ^
    --storage sqlite:///storage/optuna_studies.db
`

## 報表輸出

`powershell
python scripts/render_star_xgb_report.py ^
    --symbol BTC/USDT:USDT --timeframe 5m ^
    --strategy star_xgb_prod --dataset test ^
    --output reports/star_xgb_latest.html
`

## 即時排程

`powershell
python src/run_star_xgb_scheduler.py ^
    --strategy star_xgb_prod --symbol BTC/USDT:USDT --timeframe 5m ^
    --lookback-days 60 ^
    --params-db storage/strategy_state.db ^
    --state-db storage/strategy_state.db ^
    --exchange binanceusdm
`

- Scheduler 自動抓資料、產生訊號、並更新 runtime 狀態。
- 可串接 Discord / Slack 等通知，只要設定 webhook。

## 資料庫一覽

| DB | 用途 | 主要欄位 | 來源 |
| --- | --- | --- | --- |
| storage/market_data.db | OHLCV 快取 | 	s, iso_ts, open~olume | ackfill_ohlcv.py |
| storage/strategy_state.db | 策略參數 / 交易 / 績效 / runtime | strategy_params, strategy_trades, strategy_metrics, strategy_runtime | Optuna、報表、Scheduler |
| storage/optuna_*.db | Optuna Study | 	rials, 	rial_params, 	rial_values | pipeline_run_star_xgb.py |

## 開發守則

- 註解使用繁體中文；日誌使用英文，INFO 記錄重要事件、DEBUG 用於調試。
- 資料/訊號邏輯盡量採純函式以利測試與重構。
- 策略實作放入 src/strategies，並共用統一的接口。
- 中英文檔案請以 UTF-8 儲存，查看建議使用 scripts/cat_utf8.ps1 或設定 PowerShell OutputEncoding。
