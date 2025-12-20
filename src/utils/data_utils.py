import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def timeframe_to_offset(timeframe: str) -> pd.Timedelta:
    """將 CCXT 的 timeframe 轉換為 pandas Timedelta。"""
    multiplier = timeframe[:-1]
    unit = timeframe[-1]
    if not multiplier.isdigit():
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    value = int(multiplier)
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    if unit == "w":
        return pd.Timedelta(weeks=value)
    raise ValueError(f"Unsupported timeframe unit: {timeframe}")


def prepare_ohlcv_frame(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """清理 OHLCV 資料：排序、去重、補足缺口並轉回時間索引。"""
    if df.empty:
        return df.copy()
    frame = df.copy()
    if "timestamp" not in frame.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp")

    duplicate_count = frame.duplicated(subset="timestamp").sum()
    if duplicate_count:
        LOGGER.warning(
            "Detected %s duplicate candles, keeping the latest values", duplicate_count
        )
        frame = frame.drop_duplicates(subset="timestamp", keep="last")

    offset = timeframe_to_offset(timeframe)
    start = frame["timestamp"].iloc[0]
    end = frame["timestamp"].iloc[-1]
    expected_index = pd.date_range(start=start, end=end, freq=offset)
    frame = frame.set_index("timestamp").reindex(expected_index)
    frame.index.name = "timestamp"

    missing_count = frame["close"].isna().sum()
    if missing_count:
        LOGGER.warning(
            "Detected %s missing candles for timeframe %s; filling forward to maintain continuity",
            int(missing_count),
            timeframe,
        )
        price_cols = [
            col for col in ["open", "high", "low", "close"] if col in frame.columns
        ]
        frame[price_cols] = frame[price_cols].ffill()
        if "volume" in frame.columns:
            frame["volume"] = frame["volume"].fillna(0.0)

    frame = (
        frame.dropna(subset=["close"])
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )
    return frame


def dataframe_to_rows(df: pd.DataFrame) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        ts = int(timestamp.timestamp() * 1000)
        iso_ts = timestamp.isoformat().replace("+00:00", "Z")
        rows.append(
            (
                ts,
                iso_ts,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )
        )
    return rows



def filter_by_time(df: pd.DataFrame, column: str, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> pd.DataFrame:
    """Filter DataFrame by a datetime column between start_ts and end_ts (inclusive)."""
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


__all__ = ["prepare_ohlcv_frame", "timeframe_to_offset", "dataframe_to_rows", "filter_by_time"]
