# QuantTradSys

加密 / 股票日內交易策略實驗專案。目前維護兩套主要策略：

- `mean_reversion`：均值回歸指標組合。
- `star_xgb`：星型 K 棒辨識搭配 XGBoost 最佳化門檻。

整體流程涵蓋資料回補、Optuna 調參、回測報告與即時訊號排程。

> **UI 備註**  
> 原 FastAPI / Gradio Dashboard 已移至 `fullstack-end-backup`。`main` 僅保留策略、報表與排程相關模組。

## 目錄概覽

```
QuantTradSys/
├─ scripts/                     # 命令列入口（執行前先設定 PYTHONPATH=src）
│  ├─ backfill_ohlcv.py         # CCXT 抓取 / 回補資料（含 iso_ts 欄位）
│  ├─ pipeline_run_mean_rev.py  # 均值回歸 Optuna 管線
│  ├─ pipeline_run_star_xgb.py  # star_xgb Optuna 管線
│  ├─ render_mean_reversion_report.py
│  ├─ render_star_xgb_report.py
│  └─ ...（其他輔助腳本）
├─ src/
│  ├─ data_pipeline/            # CCXT 與 SQLite I/O，自動維護 iso_ts
│  ├─ optimization/             # Optuna 搜尋與 nested split 邏輯
│  ├─ persistence/              # 參數 / 交易 / 績效 / 狀態 儲存
│  ├─ strategies/
│  │  ├─ mean_reversion/        # 均值回歸策略
│  │  └─ star_xgb/              # 星型 K 棒 + XGBoost 策略
│  ├─ reporting/                # 報表組件
│  └─ run_*_scheduler.py        # 即時排程器
└─ storage/                     # SQLite、報表輸出、Optuna DB
```

## 安裝與環境

```powershell
python -m venv .venv
. .venv/Scripts/Activate
pip install -r requirements.txt
$env:PYTHONPATH = 'src'         # 執行 scripts/* 前必須設定
```

## 資料回補

```powershell
python scripts/backfill_ohlcv.py BTC/USD 5m 365 ^
    --db storage/market_data.db --exchange binance --prune
```

- 初次執行會建立 `storage/market_data.db` 及 `ohlcv` 資料表。  
- 每筆資料同時寫入 `ts`（毫秒）與 `iso_ts`（UTC ISO8601），方便人工檢視。  
- `--prune` 會刪除觀察窗以外的舊資料，二次執行只補缺口。
- Note: as of 2025-10 the default pair is BTC/USD. If storage/market_data.db still contains legacy BTC/USDT rows, remove the old files under storage/, backfill data again, retrain strategies, and refresh runtime state.

## Optuna 調參

### 均值回歸

```powershell
python scripts/pipeline_run_mean_rev.py ^
    --symbol BTC/USD --timeframe 5m ^
    --lookback-days 400 ^
    --n-trials 200 ^
    --study-name mean_rev_prod ^
    --storage sqlite:///storage/optuna_mean_rev.db
```

### 星型 K 棒 + XGBoost

```powershell
python scripts/pipeline_run_star_xgb.py ^
    --symbol BTC/USD --timeframe 5m ^
    --lookback-days 360 --test-days 30 ^
    --n-trials 80 ^
    --study-name star_xgb_prod ^
    --storage sqlite:///storage/optuna_studies.db
```

- 內建 nested split：外層 Train/Test 為 11 個月 / 1 個月；內層 Train/Validation 專供 Optuna 評分。  
- 調參階段自動套用 0.1% 交易成本，避免模型透過大量微利交易取得不合理報酬。  
- 最佳 Trial 會重新訓練並回測，結果寫入 `storage/strategy_state.db`。

## 報表輸出

```powershell
python scripts/render_star_xgb_report.py ^
    --symbol BTC/USD --timeframe 5m ^
    --strategy star_xgb_prod --dataset test ^
    --output reports/star_xgb_latest.html
```

```powershell
python scripts/render_mean_reversion_report.py ^
    --ohlcv-db storage/market_data.db ^
    --trades-db storage/strategy_state.db ^
    --metrics-db storage/strategy_state.db ^
    --symbol BTC/USD --timeframe 5m ^
    --start 2025-01-01T00:00Z --end 2025-12-31T23:55Z ^
    --output reports/mean_rev_latest.html
```

- 會整合交易標記、資金曲線、統計表與逐筆交易。  
- 若資料庫缺少指定策略/期間資料，腳本會自動重新回測。  
- 生成 HTML 後可直接用瀏覽器開啟或分享。

## 即時排程

```powershell
python src/run_star_xgb_scheduler.py ^
    --strategy star_xgb_prod --symbol BTC/USD --timeframe 5m ^
    --lookback-days 60 ^
    --params-db storage/strategy_state.db ^
    --state-db storage/strategy_state.db
```

```powershell
python src/run_mean_reversion_scheduler.py ^
    --strategy mean_rev_prod --symbol BTC/USD --timeframe 5m ^
    --lookback-days 400 --interval-minutes 5
```

- Scheduler 會載入參數、抓取最新資料、產生訊號並更新執行狀態。  
- 若需推播通知（Discord / Slack 等），請先設定 webhook。

## 主要資料庫表格

| 資料庫 | 用途 | 重要欄位 | 來源腳本 |
| --- | --- | --- | --- |
| `storage/market_data.db` | OHLCV 快取 | `ts`, `iso_ts`, `open`~`volume` | `backfill_ohlcv.py`、排程、報表 |
| `storage/strategy_state.db` | 策略參數 / 交易 / 績效 / 狀態 | `strategy_params`, `strategy_trades`, `strategy_metrics`, `strategy_runtime` | Optuna、報表、排程 |
| `storage/optuna_*.db` | Optuna Study | `trials`, `trial_params`, `trial_values` | `pipeline_run_*` |

## 開發指引

- 註解採用繁體中文；日誌使用英文，INFO 紀錄重要事件，DEBUG 提供調試細節。  
- 資料處理與訊號邏輯盡量採函數式撰寫，方便測試與重複利用。  
- 新增策略時，請放入 `/strategies` 並繼承既有抽象結構。  
- 中文檔案建議使用 `scripts/cat_utf8.ps1` 或設定 PowerShell `OutputEncoding` 以避免亂碼。  
- 若需恢復互動式 UI，可切換至 `fullstack-end-backup` 分支或自行 cherry-pick。
