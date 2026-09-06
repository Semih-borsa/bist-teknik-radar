#!/usr/bin/env python3
"""Nadaraya-Watson endpoint taraması.

Mevcut scanner.py akışına dokunmadan çalışır ve feed.json içine ayrı bir
"nadaraya" bölümü ekler. Hesap, LuxAlgo örneğindeki repaint kapalı endpoint
yönteminin matematiksel karşılığıdır; kaynak kod kopyalanmamıştır.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TICKERS_PATH = ROOT / "tickers.txt"
FEED_PATH = ROOT / "output" / "feed.json"

NWE_BANDWIDTH = 8.0
NWE_MULTIPLIER = 3.0
NWE_LOOKBACK = 500
NWE_MAE_LENGTH = 499
RSI_LENGTH = 14
ADX_LENGTH = 14
MIN_NWE_BARS = NWE_LOOKBACK + NWE_MAE_LENGTH - 1

SIGNAL_LABELS = {
    "BUY": "AL",
    "BUY_CANDIDATE": "AL Adayı",
    "WATCH": "İzle",
    "SELL_APPROACH": "SAT Yaklaşıyor",
    "SELL": "SAT",
    "STRONG_SELL": "Güçlü SAT",
}

SIGNAL_RANK = {
    "BUY": 0,
    "STRONG_SELL": 1,
    "SELL": 2,
    "BUY_CANDIDATE": 3,
    "SELL_APPROACH": 4,
    "WATCH": 5,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_tickers() -> list[str]:
    symbols: list[str] = []
    for line in TICKERS_PATH.read_text(encoding="utf-8").splitlines():
        symbol = line.strip().upper().removesuffix(".IS")
        if symbol and not symbol.startswith("#") and symbol.isalnum() and 2 <= len(symbol) <= 8:
            symbols.append(symbol)
    if len(symbols) < 100:
        raise RuntimeError("BIST sembol listesi eksik.")
    return sorted(set(symbols))


def extract_frame(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        first = set(download.columns.get_level_values(0))
        try:
            frame = download[ticker] if ticker in first else download.xs(ticker, axis=1, level=1)
        except (KeyError, ValueError):
            return pd.DataFrame()
    else:
        frame = download

    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()

    frame = frame[required].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def download_market(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    import yfinance as yf

    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for start in range(0, len(symbols), 50):
        batch = symbols[start:start + 50]
        tickers = [symbol + ".IS" for symbol in batch]
        try:
            data = yf.download(
                tickers,
                period="5y",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=True,
                timeout=30,
            )
        except Exception:
            failed.extend(batch)
            continue

        for symbol, ticker in zip(batch, tickers):
            frame = extract_frame(data, ticker)
            if len(frame) >= 25:
                frames[symbol] = frame
            else:
                failed.append(symbol)

    return frames, failed


def gaussian_weights(h: float = NWE_BANDWIDTH, lookback: int = NWE_LOOKBACK) -> np.ndarray:
    if h <= 0:
        raise ValueError("Bandwidth sıfırdan büyük olmalıdır.")
    x = np.arange(lookback, dtype=float)
    return np.exp(-((x * x) / (2.0 * h * h)))


def endpoint_nwe(
    close: np.ndarray | pd.Series,
    h: float = NWE_BANDWIDTH,
    lookback: int = NWE_LOOKBACK,
) -> np.ndarray:
    """Repaint kapalı endpoint Nadaraya-Watson tahmini.

    Her barda yalnızca o bar ve geçmiş `lookback - 1` kapanış kullanılır.
    Ağırlık paydası, Pine örneğindeki gibi sabit 0..lookback-1 ağırlık toplamıdır.
    """
    values = np.asarray(close, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < lookback:
        return out

    weights = gaussian_weights(h, lookback)
    denominator = float(weights.sum())
    convolution = np.convolve(values, weights, mode="full")[:len(values)]
    out[lookback - 1:] = convolution[lookback - 1:] / denominator
    return out


def wilder_rma(values: np.ndarray | pd.Series, length: int) -> np.ndarray:
    """TradingView RMA/Wilder yumuşatmasına yakın deterministik uygulama."""
    raw = np.asarray(values, dtype=float)
    out = np.full(len(raw), np.nan, dtype=float)
    start = None

    for index in range(length - 1, len(raw)):
        window = raw[index - length + 1:index + 1]
        if np.isfinite(window).all():
            out[index] = float(window.mean())
            start = index
            break

    if start is None:
        return out

    for index in range(start + 1, len(raw)):
        if np.isfinite(raw[index]) and np.isfinite(out[index - 1]):
            out[index] = ((out[index - 1] * (length - 1)) + raw[index]) / length

    return out


def rsi_wilder(close: np.ndarray | pd.Series, length: int = RSI_LENGTH) -> np.ndarray:
    values = np.asarray(close, dtype=float)
    delta = np.r_[np.nan, np.diff(values)]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[0] = np.nan
    loss[0] = np.nan

    avg_gain = wilder_rma(gain, length)
    avg_loss = wilder_rma(loss, length)
    out = np.full(len(values), np.nan, dtype=float)

    valid = np.isfinite(avg_gain) & np.isfinite(avg_loss)
    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    only_loss_zero = valid & (avg_loss == 0) & (avg_gain > 0)
    normal = valid & (avg_loss > 0)

    out[both_zero] = 50.0
    out[only_loss_zero] = 100.0
    rs = np.zeros(len(values), dtype=float)
    rs[normal] = avg_gain[normal] / avg_loss[normal]
    out[normal] = 100.0 - (100.0 / (1.0 + rs[normal]))
    return out


def adx_wilder(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    close: np.ndarray | pd.Series,
    length: int = ADX_LENGTH,
) -> np.ndarray:
    high_values = np.asarray(high, dtype=float)
    low_values = np.asarray(low, dtype=float)
    close_values = np.asarray(close, dtype=float)
    previous_close = np.r_[np.nan, close_values[:-1]]

    true_range = np.maximum.reduce(
        [
            high_values - low_values,
            np.abs(high_values - previous_close),
            np.abs(low_values - previous_close),
        ]
    )

    up_move = np.r_[np.nan, np.diff(high_values)]
    down_move = np.r_[np.nan, -np.diff(low_values)]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan

    atr = wilder_rma(true_range, length)
    plus_smoothed = wilder_rma(plus_dm, length)
    minus_smoothed = wilder_rma(minus_dm, length)

    plus_di = np.full(len(close_values), np.nan, dtype=float)
    minus_di = np.full(len(close_values), np.nan, dtype=float)
    valid_atr = np.isfinite(atr) & (atr > 0)
    plus_di[valid_atr] = 100.0 * plus_smoothed[valid_atr] / atr[valid_atr]
    minus_di[valid_atr] = 100.0 * minus_smoothed[valid_atr] / atr[valid_atr]

    denominator = plus_di + minus_di
    dx = np.full(len(close_values), np.nan, dtype=float)
    valid_di = np.isfinite(denominator) & (denominator > 0)
    dx[valid_di] = 100.0 * np.abs(plus_di[valid_di] - minus_di[valid_di]) / denominator[valid_di]
    dx[np.isfinite(denominator) & (denominator == 0)] = 0.0
    return wilder_rma(dx, length)


def with_nadaraya(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"].astype(float).to_numpy()

    estimate = endpoint_nwe(close)
    error = np.abs(close - estimate)
    mae = (
        pd.Series(error, index=data.index)
        .rolling(NWE_MAE_LENGTH, min_periods=NWE_MAE_LENGTH)
        .mean()
        .to_numpy()
        * NWE_MULTIPLIER
    )

    data["NWE"] = estimate
    data["NWE_MAE"] = mae
    data["NWE_UPPER"] = estimate + mae
    data["NWE_LOWER"] = estimate - mae
    data["RSI14"] = rsi_wilder(close)
    data["ADX14"] = adx_wilder(data["High"], data["Low"], data["Close"])
    return data


def classify_signal(current: pd.Series | dict, previous: pd.Series | dict) -> tuple[str | None, str]:
    close = float(current["Close"])
    high = float(current["High"])
    low = float(current["Low"])
    upper = float(current["NWE_UPPER"])
    lower = float(current["NWE_LOWER"])
    rsi = float(current["RSI14"])
    adx = float(current["ADX14"])
    previous_rsi = float(previous["RSI14"])

    width = upper - lower
    if not np.isfinite(width) or width <= 0:
        return None, ""

    channel_position = ((close - lower) / width) * 100.0
    pierced_lower = low <= lower
    recovered_inside = close > lower
    rsi_rising = rsi > previous_rsi
    rsi_falling = rsi < previous_rsi
    rejected_upper = high >= upper and close < upper

    if pierced_lower and recovered_inside and rsi <= 40 and rsi_rising and adx < 30:
        return "BUY", "Alt bant altına sarktı, kanal içine döndü; RSI yukarı dönüyor ve ADX filtresi uygun."

    if rejected_upper and rsi >= 65 and rsi_falling:
        return "STRONG_SELL", "Üst bant aşıldı ancak kapanış kanal içine döndü; RSI aşağı dönüyor."

    if channel_position >= 95 and rsi >= 65:
        return "SELL", "Fiyat kanalın %95 ve üzeri bölgesinde; RSI teyidi mevcut."

    if pierced_lower:
        return "BUY_CANDIDATE", "Fiyat gün içinde alt Nadaraya bandının altına sarktı; dönüş teyidi bekleniyor."

    if channel_position >= 90 and rsi >= 60:
        return "SELL_APPROACH", "Fiyat kanalın %90 ve üzeri bölgesinde; üst banda yaklaşık %10 veya daha az mesafe kaldı."

    if channel_position <= 10:
        return "WATCH", "Fiyat Nadaraya kanalının alt %10 bölgesinde; alt bant çevresi izleniyor."

    return None, ""


def nadaraya_snapshot(symbol: str, frame: pd.DataFrame) -> dict | None:
    if len(frame) < MIN_NWE_BARS:
        return None

    data = with_nadaraya(frame)
    required = ["NWE", "NWE_MAE", "NWE_UPPER", "NWE_LOWER", "RSI14", "ADX14"]
    usable = data.dropna(subset=required)
    if len(usable) < 2:
        return None

    current = usable.iloc[-1]
    previous = usable.iloc[-2]
    signal, reason = classify_signal(current, previous)
    if signal is None:
        return None

    upper = float(current["NWE_UPPER"])
    lower = float(current["NWE_LOWER"])
    close = float(current["Close"])
    width = upper - lower
    channel_position = ((close - lower) / width) * 100.0 if width > 0 else None

    return {
        "symbol": symbol,
        "marketDate": usable.index[-1].date().isoformat(),
        "close": round(close, 4),
        "estimate": round(float(current["NWE"]), 4),
        "upper": round(upper, 4),
        "lower": round(lower, 4),
        "mae": round(float(current["NWE_MAE"]), 4),
        "channelPosition": round(float(channel_position), 2) if channel_position is not None else None,
        "rsi14": round(float(current["RSI14"]), 2),
        "adx14": round(float(current["ADX14"]), 2),
        "signal": signal,
        "label": SIGNAL_LABELS[signal],
        "reason": reason,
    }


def scan_nadaraya(frames: dict[str, pd.DataFrame]) -> tuple[list[dict], int]:
    results: list[dict] = []
    insufficient = 0

    for symbol, frame in frames.items():
        if len(frame) < MIN_NWE_BARS:
            insufficient += 1
            continue
        item = nadaraya_snapshot(symbol, frame)
        if item is not None:
            results.append(item)

    results.sort(
        key=lambda item: (
            SIGNAL_RANK.get(item["signal"], 99),
            item["channelPosition"] if item["signal"] in {"BUY", "BUY_CANDIDATE", "WATCH"} else -item["channelPosition"],
            item["symbol"],
        )
    )
    return results, insufficient


def load_feed() -> dict:
    try:
        value = json.loads(FEED_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_feed(feed: dict) -> None:
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEED_PATH.write_text(json.dumps(feed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def attach_nadaraya(
    feed: dict,
    results: list[dict],
    *,
    available_symbols: int,
    failed_symbols: int,
    insufficient_symbols: int,
    duration_seconds: float,
) -> dict:
    updated = dict(feed)
    updated["nadarayaVersion"] = 1
    updated["nadarayaUpdatedAt"] = now_iso()
    updated["nadarayaStatus"] = "Güncel"
    updated["nadarayaConfig"] = {
        "method": "endpoint-non-repainting",
        "source": "Close",
        "bandwidth": NWE_BANDWIDTH,
        "multiplier": NWE_MULTIPLIER,
        "lookback": NWE_LOOKBACK,
        "maeLength": NWE_MAE_LENGTH,
        "rsiLength": RSI_LENGTH,
        "adxLength": ADX_LENGTH,
        "sellApproachChannelPosition": 90,
        "sellChannelPosition": 95,
    }
    updated["nadarayaStats"] = {
        "availableSymbols": available_symbols,
        "failedSymbols": failed_symbols,
        "insufficientHistorySymbols": insufficient_symbols,
        "matches": len(results),
        "durationSeconds": round(duration_seconds, 1),
    }
    updated["nadaraya"] = results
    return updated


def main() -> int:
    started = time.time()
    feed = load_feed()

    try:
        symbols = load_tickers()
        frames, failed = download_market(symbols)
        if not frames:
            raise RuntimeError("Nadaraya için piyasa verisi alınamadı.")

        results, insufficient = scan_nadaraya(frames)
        updated = attach_nadaraya(
            feed,
            results,
            available_symbols=len(frames),
            failed_symbols=len(failed),
            insufficient_symbols=insufficient,
            duration_seconds=time.time() - started,
        )
        write_feed(updated)
        print(
            f"Nadaraya: {len(frames)} hisse verisi; {len(results)} eşleşme; "
            f"{insufficient} hissede yeterli geçmiş yok."
        )
        return 0

    except Exception as exc:
        # Nadaraya ek modüldür; hata alırsa ana teknik feed'in commit edilmesini engellemez.
        if feed:
            feed["nadarayaUpdatedAt"] = now_iso()
            feed["nadarayaStatus"] = "Hata: " + str(exc)[:240]
            write_feed(feed)
        print(f"Nadaraya taraması atlandı: {exc}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
