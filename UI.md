# TradingView Lightweight Charts UI 規劃

## 1. 開發策略與調查備忘
- 參考 TradingView 官方 lightweight-charts (MIT 授權)；以 React + TypeScript + Vite 作為前端腳手架，利於組件化與型別檢查。
- 圖表資料需符合 lightweight-charts 的 `SeriesDataItem` 格式：`time` 可為 unix timestamp (秒)；`open/high/low/close` 為 number，volume 需另外建 Volume Series。
- 技術指標與交易點位以 Overlay 方式描繪：
  - CandleSeries 顯示 K 線。
  - LineSeries/HistogramSeries 顯示指標或成交量。
  - 交易點位可用 `PriceLine` 或圖示標記 (使用 `Series.createPriceLine` 或自繪圖層)。
- 前端狀態管理：使用 TanStack Query/React Query 處理 API 快取與重新整理；由 URL Query/Local state 控制策略、交易對、timeframe 等選項。
- 即時資料可延伸使用 WebSocket；若暫時未實作，先提供定期輪詢 + 手動刷新。

## 2. 後端 (FastAPI) API 調整需求
- `/api/configs` 維持策略、symbol、timeframe 列表，但補上 `display_name` 與可用資料期間 (min/max timestamp)。
- 新增/調整 API Response 格式：
  - `GET /api/candles`：
    ```json
    {
      "candles": [
        {"time": 1758566100, "open": 112300.1, "high": 112450.0, "low": 112200.0, "close": 112320.5, "volume": 84.81},
        ...
      ]
    }
    ```
    - `time` 為秒級 Unix timestamp (int)。
    - 允許 `limit` 參數 (預設 500) 與 `sort=asc|desc`；支援 `display_timeframe` 轉換。
  - `GET /api/trades`：
    ```json
    {
      "trades": [
        {
          "time": 1758570000,
          "side": "LONG",
          "price": 112345.6,
          "exit_time": 1758573600,
          "exit_price": 112980.0,
          "return": 0.0123,
          "meta": {"run_id": "...", "dataset": "test"}
        }
      ]
    }
    ```
    - `time`/`exit_time` 均為秒級 timestamp，方便前端掛 marker。
  - `GET /api/summary`：返回年度化報酬等數據；前端以表格呈現即可，維持現有欄位但 `created_at` 轉為 ISO UTC。
- 若要服務前端靜態資產，可在 FastAPI 新增 SPA Serve (e.g. `app.mount("/dashboard", StaticFiles(...))`)；或於 Nginx/Node 另外託管。

## 3. 前端 UI 介面設計
### 版面配置 (Desktop)
```
┌───────────────────────────────────────────────────────────────┐
│ Title Bar + Strategy Selector (策略 / 交易對 / timeframe / dataset) │
├───────────────────────────────────────────────────────────────┤
│ OHLC 主圖 (Lightweight Candlestick) + 交易點位標記              │
│  - 左上顯示當前策略與期間、右上顯示資料量與最後更新時間         │
├───────────────────────────────────────────────────────────────┤
│ 指標面板 (最多兩個 Tab)
│  1. 績效摘要 (表格 + Lazy load)
│  2. 交易紀錄 (DataGrid + 匯出 CSV 按鈕)
├───────────────────────────────────────────────────────────────┤
│ Footer: API latency/版本資訊/自動刷新切換                         │
└───────────────────────────────────────────────────────────────┘
```

### 行動版調整
- 控制列收合為 Drawer。
- 圖表高度調整為視窗 60%；下方切換 Tab 顯示摘要或紀錄。

## 4. 實作步驟 (未開始，供後續參考)
1. 建立 `web/lightweight-ui` (Vite + React + TS)；加入 `lightweight-charts`, `@tanstack/react-query`, `axios`, `zustand` (若需簡易狀態)。
2. 建立 API 型別定義與資料轉換 utilities (e.g. `mapCandleResponseToSeries`).
3. 開發共用元件：
   - `ChartContainer`: 包裝 lightweight chart 初始化、Resize Observer。
   - `ControlPanel`: 下拉選單 + 日期選擇器。
   - `PerformanceTable`, `TradesTable`: 使用 `mantine` 或 `mui` DataGrid。
4. 後端調整 API 輸出格式與參數，補單元測與文件。
5. 提供整合測試腳本 (Playwright 或 Vitest + Happy DOM) 驗證圖表與資料顯示。

## 5. 待決議事項
- 是否保留 Gradio 供快速 demo？若完全移除需調整 `scripts/run_ui_dashboard.py` 只啟動 FastAPI。
- 資料量大時的分頁/虛擬化策略 (後端 limit + 前端 infinite scroll)。
- 若要即時交易推播，應新增 WebSocket 或 Server-Sent Events。

---
此檔案僅為重開專案前的設計草稿，實作尚未開始。
