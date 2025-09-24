"""啟動均值回歸策略的即時訊號排程。"""
import argparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean reversion signal scheduler")
    parser.add_argument("--symbol", default="BTC/USDT", help="交易對，例如 BTC/USDT")
    parser.add_argument("--timeframe", default="5m", help="K 線時間週期")
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
    args = parser.parse_args()

    _configure_logging(args.log_file)

    def job() -> None:
        try:
            run_realtime_cycle(
                args.symbol,
                timeframe=args.timeframe,
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
        "Mean reversion scheduler started | symbol=%s timeframe=%s interval=%s min",
        args.symbol,
        args.timeframe,
        args.interval_minutes,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        LOGGER.info("Scheduler stopped")


if __name__ == "__main__":
    main()
