from __future__ import annotations

import csv
import json
from typing import Any

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from .services import (
    DEFAULT_CONFIDENCE,
    DEFAULT_WINDOWS,
    build_analysis,
    clean_windows,
    load_default_signal,
)


def parse_coverage(raw_value: str | None, default: float = DEFAULT_CONFIDENCE) -> float:
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError:
        return default

    return max(0.80, min(0.999, value))


def parse_duration(raw_value: str | None, default: float = 60.0) -> float:
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError:
        return default

    return max(10.0, min(300.0, value))


def parse_windows(raw_value: str | None) -> list[float]:
    if not raw_value:
        return DEFAULT_WINDOWS.copy()

    parts = [part.strip() for part in raw_value.split(",")]
    numbers: list[float] = []

    for part in parts:
        if not part:
            continue
        try:
            numbers.append(float(part.replace(",", ".")))
        except ValueError:
            continue

    return clean_windows(numbers)


def _reverse_windows(windows: list[int]) -> list[int]:
    return [windows[index] for index in range(len(windows) - 1, -1, -1)]


def build_context(request: HttpRequest) -> dict[str, Any]:
    # Load signal — prefer uploaded file (POST), fall back to default
    signal_values: list[float] = []
    upload_error: str = ""

    uploaded_file = (
        request.FILES.get("signal_file") if request.method == "POST" else None
    )
    if uploaded_file:
        try:
            raw_bytes: bytes = uploaded_file.read()
            # Try multiple encodings; parse inline to avoid stale-import issues
            import re as _re

            for _enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
                try:
                    _text = raw_bytes.decode(_enc)
                    _tokens = _re.findall(
                        r"[-+]?\d+[,.]\d+E[-+]?\d+", _text, flags=_re.IGNORECASE
                    )
                    _parsed = [float(t.replace(",", ".")) for t in _tokens]
                    if _parsed:
                        signal_values = _parsed
                        break
                except Exception:
                    continue
            if not signal_values:
                upload_error = "Файл не містить розпізнаних числових значень; використано дані за замовчуванням."
        except Exception as exc:
            upload_error = f"Помилка читання файлу ({type(exc).__name__}: {exc}); використано дані за замовчуванням."

    if not signal_values:
        signal_values = load_default_signal()

    coverage = parse_coverage(
        request.GET.get("p") or request.POST.get("p"), default=DEFAULT_CONFIDENCE
    )
    duration = parse_duration(
        request.GET.get("duration") or request.POST.get("duration"), default=60.0
    )
    windows_raw = request.GET.get("windows") or request.POST.get("windows")
    windows = parse_windows(windows_raw)

    analysis = build_analysis(
        signal_values=signal_values,
        total_duration_min=duration,
        coverage=coverage,
        windows=windows,
    )

    if analysis["histograms"]:
        first_hist_window = next(iter(analysis["histograms"]))
    else:
        first_hist_window = ""

    selected_hist_window = (
        request.GET.get("hist_window")
        or request.POST.get("hist_window")
        or first_hist_window
    )
    if selected_hist_window not in analysis["histograms"]:
        selected_hist_window = first_hist_window

    time_axis = [
        float(index) * float(analysis["sample_period_min"])
        for index in range(len(analysis["raw_values"]))
    ]

    return {
        "analysis": analysis,
        "window_columns_desc": _reverse_windows(analysis["windows"]),
        "controls": {
            "coverage": coverage,
            "duration": duration,
            "windows": ",".join(str(item) for item in windows),
        },
        "selected_hist_window": selected_hist_window,
        "query_string": request.GET.urlencode(),
        "upload_error": upload_error,
        "time_series_json": json.dumps(
            {
                "time": time_axis,
                "signal": analysis["raw_values"],
                "ppm": analysis["relative_ppm"],
            }
        ),
        "histograms_json": json.dumps(analysis["histograms"]),
        "table4_json": json.dumps(analysis["table4_rows"]),
        "polynomial_json": json.dumps(analysis["polynomial"]),
        "global_histogram_json": json.dumps(analysis["global_histogram"]),
        "interval_analysis_json": json.dumps(analysis["interval_analysis"]),
    }


@ensure_csrf_cookie
def index(request: HttpRequest) -> HttpResponse:
    context = build_context(request)
    return render(request, "analysis/index.html", context)


def export_csv(request: HttpRequest) -> HttpResponse:
    context = build_context(request)
    analysis = context["analysis"]

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = (
        "attachment; filename=calibrator_h4_6_methodical_analysis.csv"
    )
    response.write("\ufeff")

    writer = csv.writer(response)

    writer.writerow(["Параметр", "Значення"])
    writer.writerow(["Кількість відліків", int(analysis["raw_stats"]["count"])])
    writer.writerow(
        ["Середній період дискретизації, хв", f"{analysis['sample_period_min']:.4f}"]
    )
    writer.writerow([])

    writer.writerow(["Таблиця 1: Статистичні характеристики"])
    writer.writerow(
        ["Показник", "Для вихідного сигналу", "Для відносного відхилення (ppm)"]
    )
    for row in analysis["table1_rows"]:
        writer.writerow([row["label"], f"{row['raw']:.9f}", f"{row['ppm']:.9f}"])

    writer.writerow([])
    writer.writerow(["Таблиця 3: Поточні інтервали U0.95 (ppm)"])

    windows_desc = _reverse_windows(analysis["windows"])
    header = ["Хвилина"]
    for window in windows_desc:
        header.extend([f"{window} хв -0.95", f"{window} хв +0.95"])
    writer.writerow(header)

    for row in analysis["table3_rows"]:
        csv_row: list[str | int] = [row["minute"]]
        for window in windows_desc:
            value = row["values"].get(window)
            if value is None:
                csv_row.extend(["", ""])
            else:
                csv_row.extend([f"{value['lower']:.6f}", f"{value['upper']:.6f}"])
        writer.writerow(csv_row)

    writer.writerow([])
    writer.writerow(["Таблиця 4: Осереднена U0.95(t)"])
    writer.writerow(["Вікно, хв", "-0.95, ppm", "+0.95, ppm"])
    for row in analysis["table4_rows"]:
        writer.writerow(
            [f"{row['window']}", f"{row['lower']:.6f}", f"{row['upper']:.6f}"]
        )

    writer.writerow([])
    writer.writerow(["Поліном 3-го порядку"])
    writer.writerow(["U_low(t)", analysis["polynomial"]["formula_lower"]])
    writer.writerow(["U_up(t)", analysis["polynomial"]["formula_upper"]])

    return response
