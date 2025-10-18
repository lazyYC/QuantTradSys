"""啟動均值回歸策略的即時訊號排程。"""
import argparse
import logging
import os
import sqlite3
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from apscheduler.schedulers.blocking import BlockingScheduler

from pipelines.mean_reversion_realtime import run_realtime_cycle

LOGGER = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("storage/logs/mean_reversion_scheduler.log")


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def _load_strategy_candidates(params_db: Path) -> List[Dict[str, str]]:
    conn = sqlite3.connect(params_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT strategy, symbol, timeframe, updated_at
        FROM strategy_params
        ORDER BY updated_at DESC
        """
    ).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError("策略資料庫內沒有任何參數紀錄，請先完成一次訓練流程。")
    return [dict(row) for row in rows]


def _render_option(option: Dict[str, str]) -> str:
    updated = option.get("updated_at") or ""
    return f"{option['strategy']} | {option['symbol']} | {option['timeframe']} | {updated}".strip()


def _select_with_arrows(options: List[Dict[str, str]]) -> Dict[str, str]:
    try:
        import msvcrt  # type: ignore
    except ImportError as exc:  # pragma: no cover - Windows only
        raise RuntimeError("目前環境無法使用方向鍵選單") from exc
    if not sys.stdout.isatty():
        raise RuntimeError("目前輸出非互動式終端，無法使用方向鍵選單")

    index = 0
    while True:
        os.system("cls")
        print("請使用上下鍵選擇要使用的策略，Enter 確認，Esc 取消：\n")
        for i, opt in enumerate(options):
            prefix = ">" if i == index else " "
            print(f"{prefix} {_render_option(opt)}")
        key = msvcrt.getwch()
        if key in ("\r", "\n"):
            return options[index]
        if key == "\x1b":  # ESC
            raise KeyboardInterrupt
        if key in (chr(0), "\xe0"):
            second = msvcrt.getwch()
            if second == "H":  # Up
                index = (index - 1) % len(options)
            elif second == "P":  # Down
                index = (index + 1) % len(options)




def _select_with_numbers(options: List[Dict[str, str]]) -> Dict[str, str]:
    print("請輸入數字選擇要使用的策略：")
    for idx, opt in enumerate(options, start=1):
        print(f"  {idx}. {_render_option(opt)}")
    while True:
        choice = input("輸入序號：").strip()
        if not choice.isdigit():
            print("請輸入有效的數字。")
            continue
        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        print("序號超出範圍，請重新輸入。")


def _choose_strategy(options: List[Dict[str, str]]) -> Dict[str, str]:
    if len(options) == 1:
        return options[0]
    try:
        return _select_with_arrows(options)
    except (RuntimeError, KeyboardInterrupt):
        return _select_with_numbers(options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean reversion signal scheduler")
    parser.add_argument("--strategy", help="策略名稱，若不指定則提供選單")
    parser.add_argument("--symbol", help="交易對，預設使用策略紀錄中的值")
    parser.add_argument("--timeframe", help="K 線時間週期，預設使用策略紀錄中的值")
    parser.add_argument("--lookback-days", type=int, default=400, help="回補資料使用的天數")
    parser.add_argument("--interval-minutes", type=int, default=5, help="排程間隔分鐘數")
    parser.add_argument("--exchange", default="binance", help="CCXT 交易所 ID")
    parser.add_argument(
        "--params-db",
        type=Path,
        default=Path("storage/strategy_state.db"),
        help="策略參數儲存的 SQLite 路徑",
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        default=Path("storage/strategy_state.db"),
        help="即時狀態儲存的 SQLite 路徑",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="排程器寫入的日誌檔路徑",
    )
    parser.add_argument("--start-lag-seconds", type=float, default=1.0, help="每次排程執行前延遲秒數，避免行情尚未更新")
    args = parser.parse_args()

    _configure_logging(args.log_file)

    candidates = _load_strategy_candidates(args.params_db)
    chosen: Dict[str, str] | None = None

    if args.strategy:
        matches = [c for c in candidates if c["strategy"] == args.strategy]
        if args.symbol:
            matches = [c for c in matches if c["symbol"] == args.symbol]
        if args.timeframe:
            matches = [c for c in matches if c["timeframe"] == args.timeframe]
        if matches:
            chosen = matches[0]
        else:
            LOGGER.warning(
                "找不到符合的策略紀錄 (strategy=%s symbol=%s timeframe=%s)，將進入選單。",
                args.strategy,
                args.symbol,
                args.timeframe,
            )

    if chosen is None:
        chosen = _choose_strategy(candidates)

    strategy_key = chosen["strategy"]
    symbol = args.symbol or chosen["symbol"]
    timeframe = args.timeframe or chosen["timeframe"]

    LOGGER.info(
        "Scheduler 使用策略：%s | %s | %s (updated_at=%s)",
        strategy_key,
        symbol,
        timeframe,
        chosen.get("updated_at"),
    )

    def job() -> None:
        try:
            run_realtime_cycle(
                symbol,
                strategy=strategy_key,
                timeframe=timeframe,
                lookback_days=args.lookback_days,
                params_store_path=args.params_db,
                state_store_path=args.state_db,
                exchange_id=args.exchange,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Realtime cycle failed: %s", exc)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(job, "interval", minutes=args.interval_minutes, next_run_time=datetime.now(timezone.utc))
    LOGGER.info(
        "Mean reversion scheduler started | strategy=%s symbol=%s timeframe=%s interval=%s min",
        strategy_key,
        symbol,
        timeframe,
        args.interval_minutes,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        LOGGER.info("Scheduler stopped")


if __name__ == "__main__":
    main()
