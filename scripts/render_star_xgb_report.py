"""star_xgb 策略報表生成腳本。"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

import pandas as pd
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reporting.mean_reversion_report import (  # noqa: E402
    create_metrics_table,
    create_params_table,
    create_top_trades_table,
    create_trade_distribution_table,
)
from reporting.plotting import build_candlestick_figure, build_trade_overview_figure  # noqa: E402
from persistence.param_store import load_strategy_params  # noqa: E402
from persistence.trade_store import load_metrics, load_trades  # noqa: E402
from strategies.data_utils import prepare_ohlcv_frame  # noqa: E402
from strategies.star_xgb.backtest import backtest_star_xgb  # noqa: E402
from strategies.star_xgb.params import StarIndicatorParams, StarModelParams  # noqa: E402

TITLE_DEFAULT = "Star XGB Report"


def _parse_time_boundaries(start: str | None, end: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """解析時間區間參數。"""
    start_ts = pd.to_datetime(start, utc=True, errors="coerce") if start else None
    end_ts = pd.to_datetime(end, utc=True, errors="coerce") if end else None
    return start_ts, end_ts


def _filter_by_time(
    df: pd.DataFrame,
    column: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    """依據時間區間過濾資料。"""
    if df.empty or column not in df.columns:
        return df
    frame = df.copy()
    frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame = frame.dropna(subset=[column])
    if start_ts is not None:
        frame = frame[frame[column] >= start_ts]
    if end_ts is not None:
        frame = frame[frame[column] <= end_ts]
    return frame.reset_index(drop=True)


def _load_candles_from_csv(path: Path, *, start_ts: pd.Timestamp | None = None, end_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """從 CSV 讀取 K 線資料。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 OHLCV 檔案: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return _filter_by_time(df, "timestamp", start_ts, end_ts)


def _load_candles_from_db(
    db_path: Path,
    symbol: str,
    timeframe: str,
    *,
    lookback_days: int | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """從 SQLite 讀取 OHLCV 資料。"""
    if not db_path.exists():
        raise FileNotFoundError(f"找不到資料庫: {db_path}")
    conn = sqlite3.connect(db_path)
    clause = "symbol = ? AND timeframe = ?"
    params: List[object] = [symbol, timeframe]
    if start_ts is not None:
        clause += " AND ts >= ?"
        params.append(int(start_ts.timestamp() * 1000))
    if end_ts is not None:
        clause += " AND ts <= ?"
        params.append(int(end_ts.timestamp() * 1000))
    query = f"""
        SELECT ts, open, high, low, close, volume
        FROM ohlcv
        WHERE {clause}
        ORDER BY ts
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if df.empty:
        return df
    if lookback_days is not None and lookback_days > 0:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lookback_days)
        cutoff_ms = int(cutoff.timestamp() * 1000)
        df = df[df["ts"] >= cutoff_ms]
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"])
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[cols]
    return _filter_by_time(df, "timestamp", start_ts, end_ts)


def _load_trades_from_db(
    args: argparse.Namespace,
    *,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """載入策略交易紀錄。"""
    candidates: List[Path] = []
    if args.trades_db:
        candidates.append(Path(args.trades_db))
    if args.params_db:
        params_path = Path(args.params_db)
        if params_path not in candidates:
            candidates.append(params_path)
    for db_path in candidates:
        df = load_trades(
            db_path,
            strategy=args.strategy,
            dataset=args.dataset,
            symbol=args.symbol,
            timeframe=args.timeframe,
            run_id=args.run_id,
        )
        if not df.empty:
            trades = df.copy()
            trades = _filter_by_time(trades, "entry_time", start_ts, end_ts)
            trades = _filter_by_time(trades, "exit_time", start_ts, end_ts)
            return trades
    return pd.DataFrame()


def _load_metrics_from_db(
    args: argparse.Namespace,
    *,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """載入策略績效摘要。"""
    candidates: List[Path] = []
    if args.metrics_db:
        candidates.append(Path(args.metrics_db))
    if args.params_db:
        params_path = Path(args.params_db)
        if params_path not in candidates:
            candidates.append(params_path)
    for db_path in candidates:
        df = load_metrics(
            db_path,
            strategy=args.strategy,
            dataset=args.dataset,
            symbol=args.symbol,
            timeframe=args.timeframe,
            run_id=args.run_id,
        )
        if df.empty:
            continue
        metrics = df.copy()
        metrics = _filter_by_time(metrics, "created_at", start_ts, end_ts)
        if metrics.empty:
            continue
        return metrics.sort_values("created_at", ascending=False).head(1).reset_index(drop=True)
    return pd.DataFrame()


def _build_equity_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """根據交易紀錄推導權益曲線。"""
    if trades.empty or {"exit_time", "return"}.difference(trades.columns):
        return pd.DataFrame(columns=["timestamp", "equity"])
    closed = trades.dropna(subset=["exit_time", "return"]).copy()
    if closed.empty:
        return pd.DataFrame(columns=["timestamp", "equity"])
    closed = closed.sort_values("exit_time")
    returns = pd.to_numeric(closed["return"], errors="coerce").fillna(0.0)
    equity_values = (1 + returns).cumprod()
    return pd.DataFrame({"timestamp": closed["exit_time"], "equity": equity_values})


def _run_backtest_from_params(
    candles: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, object] | None]:
    """使用儲存參數重新運行回測，補齊交易與績效資料。"""
    if candles.empty or not args.params_db:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    record = load_strategy_params(
        Path(args.params_db),
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    if record is None or not isinstance(record.params, Mapping):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    payload = record.params
    indicator_payload = payload.get("indicator")
    model_payload = payload.get("model")
    model_path = payload.get("model_path")
    feature_columns = payload.get("feature_columns")
    class_means = payload.get("class_means")
    class_thresholds = payload.get("class_thresholds")
    if not all([indicator_payload, model_payload, model_path, feature_columns, class_means, class_thresholds]):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    try:
        indicator_params = StarIndicatorParams(**indicator_payload)
        model_params = StarModelParams(**model_payload)
    except TypeError:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    prepared = prepare_ohlcv_frame(candles, args.timeframe)
    if prepared.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    result = backtest_star_xgb(
        prepared,
        indicator_params,
        model_params,
        model_path=str(model_path),
        timeframe=args.timeframe,
        class_means=list(class_means),
        class_thresholds=dict(class_thresholds),
        feature_columns=list(feature_columns),
    )
    trades = result.trades.copy()
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    equity = result.equity_curve.copy()
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True, errors="coerce")
    metrics_df = pd.DataFrame([result.metrics]) if result.metrics else pd.DataFrame()
    return trades, equity, metrics_df, payload


def _collect_figures(
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    equity_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    params: Optional[Mapping[str, object]],
) -> List[Tuple[str, object]]:
    """組裝圖表與表格。"""
    figures: List[Tuple[str, object]] = []
    trimmed_trades = trades
    trimmed_equity = equity_df
    if not candles.empty:
        start_ts = candles["timestamp"].min()
        end_ts = candles["timestamp"].max()
        if not trades.empty and {"entry_time", "exit_time"}.issubset(trades.columns):
            trimmed_trades = trades[
                (trades["exit_time"] >= start_ts) & (trades["entry_time"] <= end_ts)
            ]
        if not equity_df.empty and {"timestamp"}.issubset(equity_df.columns):
            trimmed_equity = equity_df[
                (equity_df["timestamp"] >= start_ts) & (equity_df["timestamp"] <= end_ts)
            ]
        if not trimmed_trades.empty or (trimmed_equity is not None and not trimmed_equity.empty):
            overview = build_trade_overview_figure(candles, trimmed_trades, equity=trimmed_equity, show_markers=True)
        else:
            overview = build_candlestick_figure(candles, title="Price Overview")
        figures.append(("價格與交易", overview))
    if params:
        indicator_fig = create_params_table(params.get("indicator"))
        if indicator_fig is not None:
            figures.append(("指標參數", indicator_fig))
        model_fig = create_params_table(params.get("model"))
        if model_fig is not None:
            figures.append(("模型參數", model_fig))
    metrics_fig = create_metrics_table(metrics_df)
    if metrics_fig is not None:
        figures.append(("績效摘要", metrics_fig))
    distribution_fig = create_trade_distribution_table(trimmed_trades)
    if distribution_fig is not None:
        figures.append(("交易分布統計", distribution_fig))
    trades_table = create_top_trades_table(trimmed_trades)
    if trades_table is not None:
        figures.append(("交易紀錄", trades_table))
    return figures


def _write_html(figures: List[Tuple[str, object]], output: Path, title: str) -> None:
    """輸出 HTML 報表。"""
    parts = [
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        (
            "<style>body{font-family:Segoe UI,system-ui,sans-serif;margin:24px;}"
            "h1{margin-bottom:16px;}h2{margin:0 0 12px;}"
            "section.report-block{margin-top:56px;}section.report-block:first-of-type{margin-top:28px;}"
            "section.report-block .plotly-graph-div{max-width:100%;}</style>"
        ),
        "</head><body>",
        f"<h1>{title}</h1>",
    ]
    include_js = True
    for heading, fig in figures:
        parts.append("<section class='report-block'>")
        parts.append(f"<h2>{heading}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False))
        parts.append("</section>")
        include_js = False
    parts.append("</body></html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render star_xgb backtest report")
    parser.add_argument("--ohlcv", help="含 timestamp/open/high/low/close 的 OHLCV 檔案")
    parser.add_argument("--ohlcv-db", default="storage/market_data.db", help="儲存 OHLCV 的 SQLite 路徑")
    parser.add_argument("--lookback-days", type=int, default=365, help="讀取資料的天數視窗")
    parser.add_argument("--output", default="reports/star_xgb_report.html", help="輸出的 HTML 檔案路徑")
    parser.add_argument("--start", help="起始時間 (ISO 格式)")
    parser.add_argument("--end", help="結束時間 (ISO 格式)")
    parser.add_argument("--trades-db", default="storage/strategy_state.db", help="策略交易紀錄 SQLite 檔")
    parser.add_argument("--metrics-db", default="storage/strategy_state.db", help="策略績效摘要 SQLite 檔")
    parser.add_argument("--params-db", default="storage/strategy_state.db", help="策略參數 SQLite 檔")
    parser.add_argument("--strategy", default="star_xgb_default", help="策略名稱 (同 study name)")
    parser.add_argument("--dataset", default="all", help="資料集標籤 (train/test/all)")
    parser.add_argument("--symbol", default="BTC/USDT", help="交易對")
    parser.add_argument("--timeframe", default="5m", help="時間框架")
    parser.add_argument("--run-id", help="指定 run_id")
    parser.add_argument("--title", default=TITLE_DEFAULT, help="報表標題")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_ts, end_ts = _parse_time_boundaries(args.start, args.end)
    if start_ts and end_ts and start_ts > end_ts:
        raise SystemExit("起始時間需早於結束時間")

    candles = pd.DataFrame()
    if args.ohlcv:
        candles = _load_candles_from_csv(Path(args.ohlcv), start_ts=start_ts, end_ts=end_ts)
    if candles.empty:
        candles = _load_candles_from_db(
            Path(args.ohlcv_db),
            args.symbol,
            args.timeframe,
            lookback_days=args.lookback_days,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    if candles.empty:
        raise SystemExit("無法取得 K 線資料，請確認輸入來源。")

    trades_df = _load_trades_from_db(args, start_ts=start_ts, end_ts=end_ts)
    metrics_df = _load_metrics_from_db(args, start_ts=start_ts, end_ts=end_ts)
    equity_df = _build_equity_from_trades(trades_df)

    params_dict: Optional[Mapping[str, object]] = None
    if trades_df.empty or metrics_df.empty:
        print("交易或績效資料不足，改以儲存參數重新回測生成報表內容...")
        bt_trades, bt_equity, bt_metrics, params_from_bt = _run_backtest_from_params(candles, args)
        if trades_df.empty:
            trades_df = bt_trades
            trades_df = _filter_by_time(trades_df, "entry_time", start_ts, end_ts)
            trades_df = _filter_by_time(trades_df, "exit_time", start_ts, end_ts)
        if equity_df.empty:
            equity_df = bt_equity
            equity_df = _filter_by_time(equity_df, "timestamp", start_ts, end_ts)
        if metrics_df.empty:
            metrics_df = bt_metrics
        if params_from_bt is not None:
            params_dict = params_from_bt

    if params_dict is None and args.params_db:
        record = load_strategy_params(
            Path(args.params_db),
            strategy=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
        if record is not None and isinstance(record.params, Mapping):
            params_dict = record.params

    if metrics_df.empty and not trades_df.empty:
        # 以交易資料估算基本績效，避免報表為空。
        wins = (pd.to_numeric(trades_df["return"], errors="coerce") > 0).mean()
        total_return = pd.to_numeric(trades_df["return"], errors="coerce").add(1).prod() - 1 if not trades_df.empty else 0.0
        metrics_df = pd.DataFrame(
            [
                {
                    "trades": len(trades_df),
                    "win_rate": float(wins) if pd.notna(wins) else 0.0,
                    "total_return": float(total_return),
                }
            ]
        )

    figures = _collect_figures(candles, trades_df, equity_df, metrics_df, params_dict)
    if not figures:
        raise SystemExit("沒有可輸出的圖表或表格，請確認資料來源。")
    _write_html(figures, Path(args.output), args.title)


if __name__ == "__main__":
    main()
