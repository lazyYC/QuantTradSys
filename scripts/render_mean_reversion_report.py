"""渲染均值回歸策略報表的簡易腳本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Mapping

import pandas as pd
import numpy as np
import sqlite3
from plotly.graph_objects import Figure
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from reporting.mean_reversion_report import (  # noqa: E402
    create_params_table,
    create_metrics_table,
    create_trade_distribution_table,
    create_top_trades_table,
    rankings_to_dataframe,
)
from persistence.trade_store import load_metrics, load_trades  # noqa: E402
from reporting.plotting import (  # noqa: E402
    build_candlestick_figure,
    build_trade_overview_figure,
)
from persistence.param_store import load_strategy_params  # noqa: E402
from strategies.data_utils import prepare_ohlcv_frame  # noqa: E402
from strategies.mean_reversion import (  # noqa: E402
    MeanReversionParams,
    backtest_mean_reversion,
)
from utils.symbols import canonicalize_symbol  # noqa: E402

TITLE_DEFAULT = "Mean Reversion Report"


def main() -> None:
    args = parse_args()
    args.symbol = canonicalize_symbol(args.symbol)
    start_ts, end_ts = _parse_time_boundaries(args.start, args.end)
    if start_ts and end_ts and start_ts > end_ts:
        raise SystemExit("start 需早於 end")
    candles = pd.DataFrame()
    if args.ohlcv:
        csv_path = Path(args.ohlcv)
        if csv_path.exists():
            candles = _load_candles(csv_path, start_ts=start_ts, end_ts=end_ts)
        else:
            raise FileNotFoundError(f"找不到 OHLCV 檔案: {csv_path}")
    if candles.empty:
        candles = _load_candles_from_db(
            Path(args.ohlcv_db),
            args.symbol,
            args.timeframe,
            args.lookback_days,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    if candles.empty:
        raise SystemExit("無法從來源取得 K 線資料")
    rankings_df = (
        _load_rankings(Path(args.rankings_json))
        if args.rankings_json
        else pd.DataFrame()
    )
    equity_df = (
        _load_equity(Path(args.equity_csv)) if args.equity_csv else pd.DataFrame()
    )
    equity_df = _filter_by_time(equity_df, "timestamp", start_ts, end_ts)
    trades_df = _load_trades_from_db(args, start_ts=start_ts, end_ts=end_ts)
    metrics_df = _load_metrics_from_db(args, start_ts=start_ts, end_ts=end_ts)
    params_dict: Mapping[str, object] | None = None
    if trades_df.empty or equity_df.empty or metrics_df.empty:
        bt_trades, bt_equity, bt_metrics, params_from_bt = _run_backtest_from_params(
            candles, args
        )
        if trades_df.empty:
            trades_df = bt_trades
        if equity_df.empty:
            equity_df = bt_equity
        if metrics_df.empty:
            metrics_df = bt_metrics
        if params_from_bt is not None:
            params_dict = params_from_bt
    if params_dict is None and args.params_db and args.symbol and args.timeframe:
        record = load_strategy_params(
            Path(args.params_db),
            strategy=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
        if record is not None:
            params_dict = record.params
    if metrics_df.empty and not rankings_df.empty:
        metric_cols = [
            "annualized_return",
            "total_return",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "trades",
        ]
        metrics_df = rankings_df.iloc[[0]][
            [col for col in metric_cols if col in rankings_df.columns]
        ]
    if (
        params_dict is None
        and not rankings_df.empty
        and "params" in rankings_df.columns
    ):
        candidate = rankings_df.iloc[0]["params"]
        if isinstance(candidate, Mapping):
            params_dict = dict(candidate)

    if not metrics_df.empty and rankings_df.empty:
        rankings_df = rankings_to_dataframe(metrics_df.to_dict(orient="records"))
    figures = _collect_figures(candles, trades_df, equity_df, metrics_df, params_dict)
    if not figures:
        raise SystemExit("沒有可輸出的圖表或表格，請確認資料來源。")
    if not figures:
        raise SystemExit("沒有可輸出的圖表，請確認輸入資料。")
    _write_html(figures, Path(args.output), args.title)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render mean reversion backtest report"
    )
    parser.add_argument(
        "--ohlcv", help="含 timestamp/open/high/low/close 的 OHLCV 檔案"
    )
    parser.add_argument(
        "--ohlcv-db", default="storage/market_data.db", help="儲存 OHLCV 的 SQLite 路徑"
    )
    parser.add_argument(
        "--lookback-days", type=int, default=365, help="讀取資料的天數視窗"
    )
    parser.add_argument(
        "--output", default="reports/mean_rev.html", help="輸出的 HTML 檔案路徑"
    )
    parser.add_argument(
        "--start", help="起始時間 (ISO，例如 2025-09-01 或 2025-09-01T00:00)"
    )
    parser.add_argument("--end", help="結束時間 (ISO，例如 2025-09-30)")
    parser.add_argument("--rankings-json", help="grid search 排名結果 JSON 檔案")
    parser.add_argument("--equity-csv", help="backtest 權益曲線 CSV 檔案")
    parser.add_argument(
        "--trades-db",
        default="storage/strategy_state.db",
        help="策略交易紀錄 SQLite 檔",
    )
    parser.add_argument(
        "--metrics-db",
        default="storage/strategy_state.db",
        help="策略績效摘要 SQLite 檔",
    )
    parser.add_argument(
        "--params-db",
        default="storage/strategy_state.db",
        help="策略參數 SQLite (預設 storage/strategy_state.db)",
    )
    parser.add_argument("--strategy", default="mean_reversion_optuna", help="策略名稱")
    parser.add_argument(
        "--dataset", default="all", help="資料集標籤，例如 train/test/all"
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="交易對")
    parser.add_argument("--timeframe", default="5m", help="時間框架")
    parser.add_argument("--run-id", help="特定 run_id")
    parser.add_argument("--title", default=TITLE_DEFAULT, help="報表標題")
    return parser.parse_args()


if __name__ == "__main__":
    main()


def _load_rankings(path: Path | None) -> pd.DataFrame:
    """讀取排名 JSON 並轉換為 DataFrame。"""
    if path is None:
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    return rankings_to_dataframe(data)


def _load_equity(path: Path | None) -> pd.DataFrame:
    """讀取權益曲線檔案。"""
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _parse_time_boundaries(
    start: str | None, end: str | None
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_ts = pd.to_datetime(start, utc=True, errors="coerce") if start else None
    end_ts = pd.to_datetime(end, utc=True, errors="coerce") if end else None
    return start_ts, end_ts


def _filter_by_time(
    df: pd.DataFrame,
    column: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    filtered = df
    if start_ts is not None:
        filtered = filtered[filtered[column] >= start_ts]
    if end_ts is not None:
        filtered = filtered[filtered[column] <= end_ts]
    return filtered.reset_index(drop=True)


def _load_candles_from_db(
    db_path: Path,
    symbol: str,
    timeframe: str,
    lookback_days: int | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """從 SQLite 讀取 OHLCV 資料。"""
    if not db_path.exists():
        raise FileNotFoundError(f"找不到資料庫: {db_path}")
    conn = sqlite3.connect(db_path)
    clause = "symbol = ? AND timeframe = ?"
    params: list[object] = [symbol, timeframe]
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
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(
            days=lookback_days
        )
        cutoff_ms = int(cutoff.timestamp() * 1000)
        df = df[df["ts"] >= cutoff_ms]
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"])
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[cols]


def _load_candles(
    path: Path,
    *,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """載入 OHLCV 資料並確保時間欄位格式正確。"""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return _filter_by_time(df, "timestamp", start_ts, end_ts)


def _load_trades_from_db(
    args: argparse.Namespace,
    *,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """透過 trade store 載入交易紀錄。"""
    db_path: Path | None = None
    if args.trades_db:
        db_path = Path(args.trades_db)
    elif args.params_db:
        db_path = Path(args.params_db)
    if db_path is None:
        return pd.DataFrame()
    df = load_trades(
        db_path,
        strategy=args.strategy,
        dataset=args.dataset,
        symbol=args.symbol,
        timeframe=args.timeframe,
        run_id=args.run_id,
    )
    if df.empty and args.trades_db and args.params_db:
        params_path = Path(args.params_db)
        if params_path.resolve() != db_path.resolve():
            df = load_trades(
                params_path,
                strategy=args.strategy,
                dataset=args.dataset,
                symbol=args.symbol,
                timeframe=args.timeframe,
                run_id=args.run_id,
            )
    return df


def _load_metrics_from_db(
    args: argparse.Namespace,
    *,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """載入 summary metrics 用於表格或圖表。"""
    db_path: Path | None = None
    if args.metrics_db:
        db_path = Path(args.metrics_db)
    elif args.params_db:
        db_path = Path(args.params_db)
    if db_path is None:
        return pd.DataFrame()
    df = load_metrics(
        db_path,
        strategy=args.strategy,
        dataset=args.dataset,
        symbol=args.symbol,
        timeframe=args.timeframe,
        run_id=args.run_id,
    )
    if df.empty and args.metrics_db and args.params_db:
        params_path = Path(args.params_db)
        if params_path.resolve() != db_path.resolve():
            df = load_metrics(
                params_path,
                strategy=args.strategy,
                dataset=args.dataset,
                symbol=args.symbol,
                timeframe=args.timeframe,
                run_id=args.run_id,
            )
    return df


def _run_backtest_from_params(
    candles: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, object] | None]:
    """若 trade/metrics 缺失時，從參數庫重建回測結果。"""
    if candles.empty or not args.params_db or not args.symbol or not args.timeframe:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    record = load_strategy_params(
        Path(args.params_db),
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    if record is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    try:
        params = MeanReversionParams(**record.params)
    except TypeError:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    prepared = prepare_ohlcv_frame(candles, args.timeframe)
    if prepared.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    result = backtest_mean_reversion(prepared, params)
    trades = result.trades.copy()
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(
            trades["entry_time"], utc=True, errors="coerce"
        )
        trades["exit_time"] = pd.to_datetime(
            trades["exit_time"], utc=True, errors="coerce"
        )
    equity = result.equity_curve.copy()
    if not equity.empty and "timestamp" in equity.columns:
        equity["timestamp"] = pd.to_datetime(
            equity["timestamp"], utc=True, errors="coerce"
        )
    metrics = pd.DataFrame([result.metrics | {"params": record.params}])
    return trades, equity, metrics, record.params


def _filter_trades_window(
    trades: pd.DataFrame, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None
) -> pd.DataFrame:
    if trades.empty or "entry_time" not in trades.columns:
        return trades
    filtered = trades.copy()
    filtered["entry_time"] = pd.to_datetime(
        filtered["entry_time"], utc=True, errors="coerce"
    )
    filtered["exit_time"] = pd.to_datetime(
        filtered.get("exit_time"), utc=True, errors="coerce"
    )
    mask = pd.Series(True, index=filtered.index)
    if start_ts is not None:
        mask &= filtered["exit_time"].isna() | (filtered["exit_time"] >= start_ts)
    if end_ts is not None:
        mask &= filtered["entry_time"] <= end_ts
    return filtered.loc[mask].reset_index(drop=True)


def _compute_window_metrics(
    candles: pd.DataFrame, equity_df: pd.DataFrame, trades: pd.DataFrame
) -> pd.DataFrame:
    if candles.empty:
        return pd.DataFrame()
    start_ts = pd.to_datetime(candles["timestamp"]).min()
    end_ts = pd.to_datetime(candles["timestamp"]).max()
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts == end_ts:
        return pd.DataFrame()
    window_days = max((end_ts - start_ts).total_seconds() / 86400, 1 / 288)
    equity = equity_df.copy() if not equity_df.empty else pd.DataFrame()
    if not equity.empty:
        equity = equity.dropna(subset=["equity", "timestamp"]).copy()
        equity["timestamp"] = pd.to_datetime(
            equity["timestamp"], utc=True, errors="coerce"
        )
        equity = equity.dropna(subset=["timestamp"]).sort_values("timestamp")
        equity = equity.set_index("timestamp")
    metrics: dict[str, float] = {}
    if not equity.empty:
        eq_values = equity["equity"]
        total_return = eq_values.iloc[-1] / eq_values.iloc[0] - 1
        annualized = (
            (1 + total_return) ** (365 / window_days) - 1
            if window_days > 0
            else total_return
        )
        daily_curve = eq_values.resample("1D").last().ffill()
        daily_returns = daily_curve.pct_change().dropna()
        sharpe = 0.0
        if not daily_returns.empty and daily_returns.std(ddof=0) > 0:
            sharpe = daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(365)
        running_max = eq_values.cummax()
        drawdown = (eq_values / running_max - 1).min() if not eq_values.empty else 0.0
        metrics.update(
            {
                "total_return": float(total_return),
                "annualized_return": float(annualized),
                "sharpe": float(sharpe),
                "max_drawdown": float(abs(drawdown)),
            }
        )
    else:
        metrics.update(
            {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }
        )
    if not trades.empty and "return" in trades.columns:
        wins = pd.to_numeric(trades["return"], errors="coerce") > 0
        metrics["win_rate"] = float(wins.mean()) if not trades.empty else 0.0
        metrics["trades"] = int(len(trades))
    else:
        metrics["win_rate"] = 0.0
        metrics["trades"] = int(len(trades))
    return pd.DataFrame([metrics])


def _collect_figures(
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    equity_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    params: Mapping[str, object] | None,
) -> List[Tuple[str, Figure]]:
    """組裝所有要輸出的圖表/表格。"""
    figures: List[Tuple[str, Figure]] = []
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
                (equity_df["timestamp"] >= start_ts)
                & (equity_df["timestamp"] <= end_ts)
            ]
        if not trimmed_trades.empty or (
            trimmed_equity is not None and not trimmed_equity.empty
        ):
            overview = build_trade_overview_figure(
                candles,
                trimmed_trades,
                equity=trimmed_equity,
                show_markers=True,
            )
        else:
            overview = build_candlestick_figure(candles, title="Price Overview")
        figures.append(("價格與交易", overview))
    params_fig = create_params_table(params)
    if params_fig is not None:
        figures.append(("策略參數", params_fig))
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


def _write_html(figures: List[Tuple[str, Figure]], output: Path, title: str) -> None:
    """輸出簡易 HTML 報表。"""
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
        parts.append(
            fig.to_html(
                full_html=False, include_plotlyjs="cdn" if include_js else False
            )
        )
        parts.append("</section>")
        include_js = False
    parts.append("</body></html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
