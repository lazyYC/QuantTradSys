# QuantTradSys

star_xgb 為核心的加密量化策略專案，涵蓋資料擷取、Optuna 調參、報表輸出與即時排程。

## 目錄結構

`
QuantTradSys/
├── scripts/
│   ├── backfill_ohlcv.py   
│   ├── train.py   
│   ├── report.py
│   ├── run_scheduler.py
│   └── ...                       
├── src/
│   ├── data_pipeline/
│   ├── optimization/
│   ├── persistence/
│   ├── strategies/
│   ├── reporting/
│   └── ...
├── storage/
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

- 指定 `--strategy` 與 `--study` 以載入對應的訓練參數與模型。
- Scheduler 自動抓資料、產生訊號、並更新 runtime 狀態。

