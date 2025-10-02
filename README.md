# QuantTradSys

量化加密資產的均值回歸實驗專案。核心流程涵蓋：

- CCXT 取得一年期 5 分鐘 K 線並寫入 SQLite。
- Optuna 搜尋均值回歸參數，產生訓練/測試績效。
- 依指定參數重新回測並輸出 HTML 報表（含交易圖形、統計表）。
- 以排程器檢查最新資料、對照儲存參數產生即時訊號。

> **UI / FastAPI 備註**  
> 互動式 Dashboard、Gradio、FastAPI 等完整前端已移至 `fullstack-end-backup` 分支保留。`main` 僅維持報表與排程必要組件。

## 目錄概覽

```
QuantTradSys/
├─ scripts/                 # 命令列入口（需設定 PYTHONPATH=src）
│  ├─ backfill_ohlcv.py     # CCXT 抓取/回補一年期資料
│  ├─ pipeline_run_mean_rev.py  # Optuna 均值回歸調參
│  ├─ render_mean_reversion_report.py  # 產生互動式報表
│  └─ run_ui_dashboard.py   # (備份) UI 入口，僅在 fullstack-end-backup
├─ src/
│  ├─ data_pipeline/        # CCXT 與資料庫存取模組
│  ├─ optimization/         # Optuna 搜尋與切分流程
│  ├─ pipelines/            # 均值回歸回測 / 即時管線
│  ├─ persistence/          # 統一的 SQLite I/O（參數、交易、狀態）
│  ├─ reporting/            # 報表用 Table Builder
│  ├─ strategies/           # `mean_reversion` 策略實作
│  └─ run_mean_reversion_scheduler.py # 即時訊號排程入口
└─ storage/                 # SQLite 與 log / Optuna 設定
```

## 安裝與環境

```powershell
python -m venv .venv
. .venv/Scripts/Activate
pip install -r requirements.txt
$env:PYTHONPATH = 'src'   # 執行 scripts/* 前請先設定
```

## 資料準備：回補一年 K 線

```powershell
python scripts/backfill_ohlcv.py BTC/USDT 5m 365 \
    --db storage/market_data.db --exchange binance --prune
```
- 初次執行會建立 `storage/market_data.db` 並建表。
- 再次執行僅抓取缺口，`--prune` 會清理視窗前舊資料。

## 參數搜尋與結果儲存

```powershell
python scripts/pipeline_run_mean_rev.py \
    --symbol BTC/USDT --timeframe 5m --lookback-days 400 \
    --n-trials 200 --study-name mean_rev_demo \
    --storage sqlite:///storage/optuna_mean_rev.db
```
- 成功後會在 `storage/strategy_state.db` 寫入最佳參數 (`strategy_params`)、交易紀錄 (`strategy_trades`) 與績效摘要 (`strategy_metrics`)。

## 報表輸出

```powershell
python scripts/render_mean_reversion_report.py \
    --ohlcv-db storage/market_data.db \
    --trades-db storage/strategy_state.db \
    --metrics-db storage/strategy_state.db \
    --symbol BTC/USDT --timeframe 5m \
    --start 2024-01-01T00:00Z --end 2024-12-31T23:55Z \
    --output reports/mean_rev_latest.html
```
- 主圖：K 線 + 紅/綠進出場標記 + 資金曲線。
- 子圖：每筆交易報酬柱狀圖（共用 X 軸）。
- 表格：策略參數、視窗內績效摘要、交易分布統計、逐筆交易列表。

## 即時訊號排程

1. 確保已有最佳參數寫入 `strategy_state.db`。
2. 設定 `.env`（如 Discord Webhook）。
3. 啟動排程器：
   ```powershell
   python src/run_mean_reversion_scheduler.py \
       --lookback-days 400 --interval-minutes 5 \
       --params-db storage/strategy_state.db \
       --state-db storage/strategy_state.db
   ```
- Scheduler 會抓最新資料 → 對照參數 → 產生訊號 → 更新 runtime state。

## 常見資料表

| 資料庫 | 目的 | 關聯腳本 |
| --- | --- | --- |
| `storage/market_data.db` | OHLCV 快取 | `backfill_ohlcv.py`, 報表、排程 |
| `storage/strategy_state.db` | 最佳參數、回測交易、報表統計 | Optuna、報表、排程 |
| `storage/optuna_mean_rev.db` | Optuna Study 記錄 | `pipeline_run_mean_rev.py` |

## UI / FastAPI 備份

- 分支：`fullstack-end-backup`
- 內容：FastAPI 路由 (`src/ui/server.py`)、Gradio 介面、Dashboard Scripts。
- 若需回復互動式 UI，請切換至該分支或 cherry-pick 相關提交。

## 開發建議

- 所有模組預設以函數式流程組合，便於單元測試與重複使用。
- 重要日誌以英文 INFO/DEBUG 記錄，方便整合第三方監控。
- 建議透過虛擬環境或容器執行，確保 `PYTHONPATH` 與 SQLite 路徑一致。
