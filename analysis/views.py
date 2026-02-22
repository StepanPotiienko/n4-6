from __future__ import annotations

import csv
import json
from typing import Any

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from .services import (
    DEFAULT_WINDOWS,
    MODEL_LABELS,
    MODEL_ORDER,
    build_analysis,
    clean_windows,
    load_default_signal,
)

MODEL_COLORS = {
    "quantile": "#1254ff",
    "normal": "#f85f2a",
    "robust": "#00a779",
}


def parse_coverage(raw_value: str | None, default: float = 0.95) -> float:
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


def parse_windows(raw_value: str | None) -> list[int]:
    if not raw_value:
        return DEFAULT_WINDOWS.copy()

    parts = [part.strip() for part in raw_value.split(",")]
    numbers: list[int] = []

    for part in parts:
        if not part:
            continue
        if not part.isdigit():
            continue
        numbers.append(int(part))

    return clean_windows(numbers)


def build_context(request: HttpRequest) -> dict[str, Any]:
    signal_values = load_default_signal()

    coverage = parse_coverage(request.GET.get("p"), default=0.95)
    duration = parse_duration(request.GET.get("duration"), default=60.0)
    windows = parse_windows(request.GET.get("windows"))

    analysis = build_analysis(
        signal_values=signal_values,
        total_duration_min=duration,
        coverage=coverage,
        windows=windows,
    )

    if analysis["histograms"]:
        first_hist_window = next(iter(analysis["histograms"]))
    else:
        first_hist_window = "1"

    selected_hist_window = request.GET.get("hist_window", first_hist_window)
    if selected_hist_window not in analysis["histograms"]:
        selected_hist_window = first_hist_window

    time_axis = [
        float(index) * float(analysis["sample_period_min"])
        for index in range(len(analysis["raw_values"]))
    ]

    winner_model = analysis["competition"][0]["model"] if analysis["competition"] else MODEL_ORDER[0]
    winner_score = analysis["competition"][0]["score"] if analysis["competition"] else 0.0

    return {
        "analysis": analysis,
        "window_rows": analysis["window_rows"],
        "competition": analysis["competition"],
        "winner_model": winner_model,
        "winner_score": winner_score,
        "model_order": MODEL_ORDER,
        "model_labels": MODEL_LABELS,
        "model_colors": MODEL_COLORS,
        "controls": {
            "coverage": coverage,
            "duration": duration,
            "windows": ",".join(str(item) for item in windows),
        },
        "selected_hist_window": selected_hist_window,
        "query_string": request.GET.urlencode(),
        "time_series_json": json.dumps(
            {
                "time": time_axis,
                "signal": analysis["raw_values"],
                "ppm": analysis["relative_ppm"],
            }
        ),
        "model_series_json": json.dumps(analysis["model_series"]),
        "competition_json": json.dumps(analysis["competition"]),
        "histograms_json": json.dumps(analysis["histograms"]),
        "polynomials_json": json.dumps(analysis["polynomials"]),
        "model_labels_json": json.dumps(MODEL_LABELS),
        "model_colors_json": json.dumps(MODEL_COLORS),
    }


def index(request: HttpRequest) -> HttpResponse:
    context = build_context(request)
    return render(request, "analysis/index.html", context)


def export_csv(request: HttpRequest) -> HttpResponse:
    context = build_context(request)
    analysis = context["analysis"]

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = "attachment; filename=calibrator_h4_6_analysis.csv"
    response.write("\ufeff")

    writer = csv.writer(response)

    writer.writerow(["Параметр", "Значення"])
    writer.writerow(["Кількість відліків", int(analysis["raw_stats"]["count"])])
    writer.writerow(["Середній період дискретизації, хв", f"{analysis['sample_period_min']:.4f}"])
    writer.writerow(["Середнє значення сигналу", f"{analysis['raw_stats']['mean']:.9f}"])
    writer.writerow([])

    writer.writerow(
        [
            "Вікно, хв",
            "Модель",
            "Нижня межа U0.95, ppm",
            "Верхня межа U0.95, ppm",
            "Ширина інтервалу, ppm",
            "Внутрішнє покриття",
        ]
    )

    for row in analysis["window_rows"]:
        for model in MODEL_ORDER:
            model_row = row["models"][model]
            writer.writerow(
                [
                    row["window"],
                    MODEL_LABELS[model],
                    f"{model_row['lower']:.6f}",
                    f"{model_row['upper']:.6f}",
                    f"{model_row['width']:.6f}",
                    f"{model_row['in_sample_coverage']:.4f}",
                ]
            )

    writer.writerow([])
    writer.writerow(["Рейтинг моделей"])
    writer.writerow(["Місце", "Модель", "Похибка покриття", "Середня ширина", "Стабільність", "Score"])

    for row in analysis["competition"]:
        writer.writerow(
            [
                row["rank"],
                row["label"],
                f"{row['coverage_error']:.6f}",
                f"{row['mean_width']:.6f}",
                f"{row['stability']:.6f}",
                f"{row['score']:.6f}",
            ]
        )

    return response
