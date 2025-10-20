# Memo — 2025-10-03 會話紀錄

## 背景
- 主幹 (`main`) 已移除 FastAPI + Gradio UI；保留報表與排程功能。
- `fullstack-end-backup` 分支保存完整 UI 堆疊，隨時可回溯。

## 對話摘要（逐題紀錄）

### 1. 單一參數組合回測是否已實作？
- **使用者：** 想驗證固定參數在延伸時段的回測結果，並與即時訊號比對，詢問 reporting 是否已提供。
- **回覆：** `scripts/render_mean_reversion_report.py` 的 `_run_backtest_from_params` 會從參數庫載入最新 `MeanReversionParams`，在無交易/績效記錄時自動重新跑 `backtest_mean_reversion`。
  - 搭配 `--start/--end` 可指定新的時間窗口。
  - 建議流程：
    1. 先用 `scripts/backfill_ohlcv.py` 更新 `storage/market_data.db`。
    2. 執行
       ```powershell
       $env:PYTHONPATH='src'
       python scripts/render_mean_reversion_report.py \
           --ohlcv-db storage/market_data.db \
           --trades-db storage/strategy_state.db \
           --metrics-db storage/strategy_state.db \
           --symbol BTC/USD --timeframe 5m \
           --start 2025-01-01T00:00Z --end 2025-01-07T23:55Z \
           --output reports/mean_rev_latest.html
       ```
  - 由此即可確認「固定參數 + 新資料」下的交易/訊號是否與排程一致。

### 2. 開發新策略（含 ML）需要怎麼做？
- **使用者：** 想在現有架構下開發新的 ML 策略，但不想大改抽象層，詢問是否直接複製平行實作即可。
- **回覆建議：**
  1. **策略程式**：在 `src/strategies/your_strategy/` 建立資料整備、回測 (`backtest_your_strategy`)、即時決策 (`generate_realtime_decision`) 等，回傳結構比照 `MeanReversionBacktestResult`。
  2. **管線與 CLI**：
     - 新增 `src/pipelines/your_strategy.py`（含訓練/驗證/ML 流程）。
     - 建立 `scripts/pipeline_run_your_strategy.py` 呼叫上述流程，並透過 `persistence.param_store`、`persistence.trade_store` 寫入結果。
  3. **報表**：
     - 可複製 `scripts/render_mean_reversion_report.py` 為策略專用版本，只需調整 import / 標題。
  4. **即時排程**：
     - 複製 `src/run_mean_reversion_scheduler.py`，改成載入新策略的參數與決策函式。
  5. **架構選擇**：若尚未需要抽象層，可先採「平行複製」與 mean reversion 共存；待策略數增多，再考慮提煉共用介面。

## 重要提醒
- `__pycache__/` 為 Python 自動產生，刪除後會在下次執行再出現；可依需要手動清理。
- 報表與排程皆依賴 `storage/strategy_state.db`；確保 Optuna 或自訂回測流程有寫入結果。
- UI 相關檔案僅在 `fullstack-end-backup` 分支，主幹不再保留。
