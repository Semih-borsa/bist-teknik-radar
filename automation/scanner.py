#!/usr/bin/env python3
"""BIST teknik gözlem ve T+1/T+3/T+5 durum takibi; işlem emri üretmez."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TICKERS_PATH = ROOT / "tickers.txt"
FEED_PATH = ROOT / "output" / "feed.json"
MIN_VOLUME_RATIO = 1.2
FOLLOW_UP_OFFSETS = (1, 3, 5)
MAX_TRACKING_RECORDS = 750
TRACKING_DAYS = 190
STATUS_LABELS = {
    "CONFIRMED": "Teyit aldı",
    "PARTIAL": "Kısmi teyit",
    "NOT_CONFIRMED": "Teyit gelmedi",
    "REVERSED": "Tersine döndü",
    "WAITING": "Veri bekleniyor",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_tickers() -> list[str]:
    symbols = []
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
            data = yf.download(tickers, period="9mo", interval="1d", group_by="ticker",
                               auto_adjust=False, actions=False, progress=False,
                               threads=True, timeout=30)
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


def with_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"].astype(float)
    for length in (5, 8, 13):
        data[f"SMA{length}"] = close.rolling(length).mean()
    data["VOL20"] = data["Volume"].astype(float).rolling(20).mean()
    return data


def market_point(row: pd.Series, market_date: str) -> dict:
    volume_mean = float(row["VOL20"])
    volume_ratio = float(row["Volume"]) / volume_mean if volume_mean > 0 else None
    return {
        "marketDate": market_date,
        "close": round(float(row["Close"]), 4),
        "sma5": round(float(row["SMA5"]), 4),
        "sma8": round(float(row["SMA8"]), 4),
        "sma13": round(float(row["SMA13"]), 4),
        "volumeRatio": round(volume_ratio, 3) if volume_ratio is not None and math.isfinite(volume_ratio) else None,
    }


def observation(symbol: str, frame: pd.DataFrame) -> list[dict]:
    data = with_indicators(frame)
    usable = data.dropna(subset=["SMA5", "SMA8", "SMA13", "VOL20"])
    if len(usable) < 2:
        return []
    current, previous = usable.iloc[-1], usable.iloc[-2]
    c, p = float(current["Close"]), float(previous["Close"])
    av = [float(current[f"SMA{x}"]) for x in (5, 8, 13)]
    pav = [float(previous[f"SMA{x}"]) for x in (5, 8, 13)]
    base = {"symbol": symbol, **market_point(current, usable.index[-1].date().isoformat())}
    volume_ok = base["volumeRatio"] is not None and base["volumeRatio"] >= MIN_VOLUME_RATIO
    found = []
    ordered = av[0] > av[1] > av[2]
    if ordered and c > max(av) and p <= max(pav) and volume_ok:
        found.append({**base, "strategy": "momentum", "event": "ABOVE",
                      "reason": "Fiyat SMA 5/8/13 üzerine yeni çıktı; hacim filtresi sağlandı."})
    touched = float(current["Low"]) <= av[1] * 1.003
    recovered = c > av[1] and c > float(current["Open"])
    if ordered and touched and recovered and p > pav[2] and volume_ok:
        found.append({**base, "strategy": "pullback", "event": "RECOVERY",
                      "reason": "Olumlu ortalama diziliminde SMA 8 çevresinden toparlandı."})
    if c < min(av) and p >= min(pav):
        found.append({**base, "strategy": "weakening", "event": "BELOW",
                      "reason": "Fiyat SMA 5/8/13 altına yeni indi."})
    return found


def evaluate_status(strategy: str, point: dict) -> tuple[str, str]:
    close = point["close"]
    averages = [point["sma5"], point["sma8"], point["sma13"]]
    ordered = averages[0] > averages[1] > averages[2]
    volume = point.get("volumeRatio")
    volume_text = "hacim verisi yok" if volume is None else f"hacim oranı ×{volume:.2f}"
    if strategy == "momentum":
        if ordered and close > max(averages):
            if volume is not None and volume >= 1.0:
                return "CONFIRMED", f"Fiyat üç ortalamanın üzerinde ve olumlu dizilim korunuyor; {volume_text}."
            return "PARTIAL", f"Fiyat üç ortalamanın üzerinde kaldı ancak {volume_text}."
        if close < averages[2]:
            return "REVERSED", "Fiyat SMA 13 altına indi; ilk momentum yapısı korunmadı."
        return "NOT_CONFIRMED", "Fiyat üç ortalamanın üzerinde kalamadı; olumlu dizilim tam korunmuyor."
    if strategy == "pullback":
        if ordered and close > averages[1]:
            if volume is not None and volume >= 1.0:
                return "CONFIRMED", f"Fiyat SMA 8 üzerinde ve olumlu dizilim korunuyor; {volume_text}."
            return "PARTIAL", f"Fiyat SMA 8 üzerinde kaldı ancak {volume_text}."
        if close < averages[2]:
            return "REVERSED", "Fiyat SMA 13 altına indi; toparlanma yapısı tersine döndü."
        return "NOT_CONFIRMED", "Fiyat SMA 8 üzerinde kalamadı; toparlanma koşulu teyit edilmedi."
    if strategy == "weakening":
        if close < min(averages):
            return "CONFIRMED", "Fiyat üç ortalamanın altında kaldı; zayıflama görünümü sürüyor."
        if close < averages[1]:
            return "PARTIAL", "Fiyat SMA 8 altında ancak üç ortalamanın tamamının altında değil."
        if close > max(averages):
            return "REVERSED", "Fiyat üç ortalamanın üzerine döndü; önceki zayıflama görünümü tersine döndü."
        return "NOT_CONFIRMED", "Fiyat ortalama bölgesine döndü; zayıflama koşulu tam korunmadı."
    raise ValueError("Bilinmeyen teknik gözlem türü.")


def track_id(item: dict) -> str:
    return "|".join((item["marketDate"], item["symbol"], item["strategy"], item["event"]))


def signal_track(item: dict) -> dict:
    return {
        "id": track_id(item), "symbol": item["symbol"], "strategy": item["strategy"],
        "event": item["event"], "signalDate": item["marketDate"], "marketDate": item["marketDate"],
        "close": item["close"], "sma5": item["sma5"], "sma8": item["sma8"],
        "sma13": item["sma13"], "volumeRatio": item.get("volumeRatio"),
        "reason": item.get("reason", ""), "checks": [], "latestStatus": "WAITING",
        "latestReason": "Sonraki işlem günü verisi bekleniyor."
    }


def load_previous_feed() -> dict:
    try:
        value = json.loads(FEED_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def build_tracking(previous: dict, observations: list[dict], market_date: str) -> list[dict]:
    by_id: dict[str, dict] = {}
    for raw in previous.get("tracking", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            continue
        checks = raw.get("checks", [])
        by_id[raw["id"]] = {**raw, "checks": checks if isinstance(checks, list) else []}
    for old in previous.get("observations", []):
        if isinstance(old, dict) and all(old.get(key) is not None for key in ("symbol", "strategy", "event", "marketDate", "close", "sma5", "sma8", "sma13")):
            by_id.setdefault(track_id(old), signal_track(old))
    for item in observations:
        by_id.setdefault(track_id(item), signal_track(item))
    cutoff = (date.fromisoformat(market_date) - timedelta(days=TRACKING_DAYS)).isoformat()
    tracks = [item for item in by_id.values() if str(item.get("signalDate", "")) >= cutoff]
    tracks.sort(key=lambda item: (item.get("signalDate", ""), item.get("symbol", "")), reverse=True)
    return tracks[:MAX_TRACKING_RECORDS]


def evaluate_tracking(tracking: list[dict], frames: dict[str, pd.DataFrame]) -> tuple[list[dict], list[dict]]:
    added: list[dict] = []
    for track in tracking:
        frame = frames.get(track.get("symbol"))
        if frame is None or frame.empty:
            continue
        data = with_indicators(frame).dropna(subset=["SMA5", "SMA8", "SMA13", "VOL20"])
        dates = [value.date().isoformat() for value in data.index]
        signal_date = track.get("signalDate")
        if signal_date not in dates:
            continue
        signal_position = dates.index(signal_date)
        checks = [item for item in track.get("checks", []) if isinstance(item, dict) and item.get("offset") in FOLLOW_UP_OFFSETS]
        by_offset = {item["offset"]: item for item in checks}
        for offset in FOLLOW_UP_OFFSETS:
            target_position = signal_position + offset
            if target_position >= len(data):
                continue
            row = data.iloc[target_position]
            point = market_point(row, dates[target_position])
            status, reason = evaluate_status(track["strategy"], point)
            check = {"offset": offset, "status": status, "label": STATUS_LABELS[status], "reason": reason, **point}
            old = by_offset.get(offset)
            by_offset[offset] = check
            if old is None:
                added.append({"id": track["id"], "symbol": track["symbol"], "strategy": track["strategy"], "event": track["event"], "signalDate": signal_date, **check})
        track["checks"] = [by_offset[offset] for offset in FOLLOW_UP_OFFSETS if offset in by_offset]
        if track["checks"]:
            latest = track["checks"][-1]
            track["latestStatus"] = latest["status"]
            track["latestReason"] = latest["reason"]
        else:
            track["latestStatus"] = "WAITING"
            track["latestReason"] = "Sonraki işlem günü verisi bekleniyor."
    added.sort(key=lambda item: (item["marketDate"], item["offset"], item["symbol"]), reverse=True)
    return tracking, added


def write_feed(feed: dict) -> None:
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEED_PATH.write_text(json.dumps(feed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def telegram_message(feed: dict) -> str:
    observations, follow_ups = feed.get("observations", []), feed.get("followUps", [])
    names = {"momentum": "Momentum", "pullback": "Pullback", "weakening": "Zayıflama"}
    lines = ["📊 BIST TEKNİK GÖZLEM", f"Piyasa tarihi: {feed.get('marketDate') or '—'}",
             f"Taranan: {feed.get('scannedSymbols', 0)} hisse"]
    if observations:
        lines.append(f"\nYeni gözlem: {len(observations)}")
        for item in observations[:35]:
            lines.append(f"• {item['symbol']} · {names.get(item['strategy'], item['strategy'])} · {item['close']:.2f} TL")
        if len(observations) > 35:
            lines.append(f"… ve {len(observations) - 35} gözlem daha")
    else:
        lines.append("\nYeni teknik koşul oluşmadı.")
    if follow_ups:
        counts = {key: 0 for key in STATUS_LABELS}
        for item in follow_ups:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        lines.append("\nTakip özeti: " + " · ".join(f"{STATUS_LABELS[key]} {value}" for key, value in counts.items() if value))
        for item in follow_ups[:20]:
            lines.append(f"• {item['symbol']} T+{item['offset']} · {item['label']}")
        if len(follow_ups) > 20:
            lines.append(f"… ve {len(follow_ups) - 20} takip sonucu daha")
    lines.append("\nBilgilendirme amaçlı teknik durum takibidir; işlem emri değildir.")
    return "\n".join(lines)[:4000]


def send_telegram(message: str) -> None:
    import requests
    token, chat_id = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets tanımlı değil; bildirim atlandı.")
        return
    response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             json={"chat_id": chat_id, "text": message}, timeout=20)
    response.raise_for_status()


def main() -> int:
    started = time.time()
    try:
        symbols = load_tickers()
        frames, failed = download_market(symbols)
        if not frames:
            raise RuntimeError("Doğrulanmış piyasa verisi alınamadı.")
        observations = []
        for symbol, frame in frames.items():
            observations.extend(observation(symbol, frame))
        observations.sort(key=lambda item: (item["strategy"], -(item.get("volumeRatio") or 0), item["symbol"]))
        market_date = max(frame.index[-1].date().isoformat() for frame in frames.values())
        tracking = build_tracking(load_previous_feed(), observations, market_date)
        tracking, follow_ups = evaluate_tracking(tracking, frames)
        feed = {
            "version": 2, "trackingVersion": 1, "updatedAt": now_iso(), "marketDate": market_date,
            "status": "Güncel", "scannedSymbols": len(frames), "failedSymbols": len(failed),
            "durationSeconds": round(time.time() - started, 1), "observations": observations,
            "followUps": follow_ups, "tracking": tracking,
        }
        write_feed(feed)
        send_telegram(telegram_message(feed))
        print(f"{len(frames)} hisse tarandı; {len(observations)} yeni gözlem; {len(follow_ups)} takip sonucu.")
        return 0
    except Exception as exc:
        try:
            previous = load_previous_feed() or {"version": 2, "observations": [], "followUps": [], "tracking": []}
            previous.update({"updatedAt": now_iso(), "status": "Hata: " + str(exc)[:240]})
            write_feed(previous)
            send_telegram("⚠️ BIST teknik gözlem taraması tamamlanamadı.\n" + str(exc)[:300])
        except Exception:
            pass
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
