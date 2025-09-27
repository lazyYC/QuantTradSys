# UI Dashboard 規劃

## 目標
- 透過互動式介面檢視回測與即時執行資料。
- 在 K 線圖上標記事件：進場 / 出場時刻與價格、高低狀態。
- 支援多策略、多交易對與彈性 timeframe。
- 可在資料量大時進行分段載入，避免一次性拉取一年資料。
- 為未來的「逐筆交易清單 + 視覺化報酬條」功能預留擴充點。

## 架構概覽
```
┌────────────────────┐          ┌────────────────────────┐
│   Gradio Blocks UI ├─────────▶│  FastAPI Router (/api) │
└────────┬───────────┘          └──────────┬─────────────┘
         │                                   │
         │                                   ▼
         │                        ┌────────────────────┐
         │                        │ UI Data Service    │
         │                        │ (查詢 + 聚合邏輯) │
         │                        └────────┬───────────┘
         │                                   │
         ▼                                   ▼
 ┌────────────────┐            ┌────────────────────────────┐
 │ Plotly 視覺化   │            │ SQLite: market_data.db      │
 │ (K線 + 標記等) │            │         strategy_state.db    │
 └────────────────┘            └────────────────────────────┘
```

## 模組拆分
- `src/ui/data_service.py`
  - 封裝資料庫連線、OHLCV 查詢、時間區間篩選、timeframe 聚合。
  - 提供 trades overlay 所需的 entry/exit 座標與 metadata。
  - 以函數式設計實作，利於單元測試與重用。
- `src/ui/plotting.py`
  - 使用 Plotly 建立 K 線圖、交易標記、持倉色帶等視覺元素。
  - 預留 `render_trade_rows()` 介面，未來串接「報酬水平條」視覺化。
- `src/ui/server.py`
  - 建立 FastAPI App，提供 `/api/candles`, `/api/trades`, `/api/summary` 等端點。
  - 端點支援 `symbol`, `timeframe`, `strategy`, `start`, `end` 等查詢參數。
  - 將分頁 / 時間滑動視為一等公民設計。
- `src/ui/gradio_app.py`
  - 建立 Gradio Blocks：控制面板（策略、日期區間）、K 線圖、摘要表格。
  - 控制面板事件觸發後端 API，將結果透過 Plotly 圖形顯示。
  - 預留第二個 Tab / Accordion 作為「逐筆交易可視化」區域。
- `scripts/run_ui_dashboard.py`
  - CLI 入口：初始化 FastAPI + Gradio，支援獨立執行或指定 host/port。

## 資料流與互動
1. 使用者選擇策略 / 交易對 / timeframe / 日期範圍。
2. Gradio 呼叫 API：
   - `/api/candles` 回傳 K 線 + 指標（可擴充技術指標線）。
   - `/api/trades` 回傳該區間的交易紀錄。
   - `/api/summary` 回傳績效指標，供表格顯示。
3. Plotly 將 K 線與交易標記繪製於同一圖層，支援滑鼠拖曳 / 放大。
4. 如使用者拖曳超出範圍，前端再次發出查詢以取得新的資料片段。

## 擴充考量：逐筆交易清單 + 視覺化報酬條
- `data_service.fetch_trades()` 回傳完整欄位，Gradio UI 可直接生成 DataFrame。
- 於 `plotting.py` 新增 `build_trade_return_bars(trades_df)` 函式，將 `return` 轉為水平方向的條狀圖，搭配顏色呈現盈虧。
- `gradio_app.py` 預留一個 Tab，未來只需呼叫上述函式並將圖表內嵌即可。

## 佈署與運維
- FastAPI 與 Gradio 同程式啟動，預設 listen `http://127.0.0.1:7861`（可調整）。
- 日誌寫入 `storage/logs/ui_server.log`，INFO 記錄請求、DEBUG 追蹤快取命中。
- UI 運行過程不修改資料庫內容，僅執行 `SELECT`。
- 若資料筆數龐大，可於 `data_service` 導入簡單快取（例如 LRU）或升級為 PostgreSQL。
