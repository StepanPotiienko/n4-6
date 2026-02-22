from __future__ import annotations

import re
from pathlib import Path
from statistics import NormalDist

import numpy as np

DEFAULT_WINDOWS = [1, 2, 5, 10, 30, 60]
MODEL_ORDER = ["quantile", "normal", "robust"]
MODEL_LABELS = {
    "quantile": "Гілка A: емпіричні квантілі",
    "normal": "Гілка B: нормальна модель",
    "robust": "Гілка C: робастна MAD-модель",
}
MODEL_CRITIQUE = {
    "quantile": {
        "idea": "Не робить припущення про закон розподілу.",
        "critique": "Сильно залежить від обсягу сегмента: на коротких інтервалах межі "
        "можуть стрибати.",
    },
    "normal": {
        "idea": "Найпростіша формула: u = mu ± k·sigma.",
        "critique": "При негаусових хвостах або викидах межі можуть бути занадто оптимістичними.",
    },
    "robust": {
        "idea": "Стійка до викидів завдяки median і MAD.",
        "critique": "За малих вибірок може недооцінювати реальні хвости розподілу.",
    },
}


def parse_signal_text(raw_text: str) -> list[float]:
    tokens = re.findall(r"[-+]?\d+[,.]\d+E[-+]?\d+", raw_text, flags=re.IGNORECASE)
    return [float(token.replace(",", ".")) for token in tokens]


def load_default_signal() -> list[float]:
    file_path = Path(__file__).resolve().parents[1] / "data" / "raw_signal.txt"
    raw_text = file_path.read_text(encoding="utf-8")
    return parse_signal_text(raw_text)


def clean_windows(windows: list[int] | None) -> list[int]:
    if not windows:
        return DEFAULT_WINDOWS.copy()

    output: list[int] = []
    for window in windows:
        if window > 0 and window not in output:
            output.append(window)

    if not output:
        return DEFAULT_WINDOWS.copy()

    output.sort()
    return output


def compute_basic_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else 0.0
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    variance = float(np.var(arr, ddof=1)) if n > 1 else 0.0

    if n > 2:
        sigma_pop = float(np.std(arr, ddof=0))
        if sigma_pop > 0:
            centered = arr - mean
            m3 = float(np.mean(centered ** 3))
            m4 = float(np.mean(centered ** 4))
            skewness = m3 / (sigma_pop ** 3)
            excess = m4 / (sigma_pop ** 4) - 3.0
        else:
            skewness = 0.0
            excess = 0.0
    else:
        skewness = 0.0
        excess = 0.0

    return {
        "count": float(n),
        "mean": mean,
        "std": std,
        "variance": variance,
        "sem": float(sem),
        "skewness": float(skewness),
        "excess": float(excess),
        "min": float(np.min(arr)) if n else 0.0,
        "max": float(np.max(arr)) if n else 0.0,
        "range": float(np.max(arr) - np.min(arr)) if n else 0.0,
        "median": float(np.median(arr)) if n else 0.0,
    }


def compute_relative_ppm(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if mean == 0.0:
        return [0.0 for _ in values]
    rel = (arr - mean) / mean * 1_000_000.0
    return [float(item) for item in rel]


def split_segments(values: list[float], points_per_segment: int) -> list[list[float]]:
    if points_per_segment <= 0:
        return []

    full_count = len(values) // points_per_segment
    segments: list[list[float]] = []

    for index in range(full_count):
        start = index * points_per_segment
        end = start + points_per_segment
        segments.append(values[start:end])

    return segments


def quantile_bounds(segment: list[float], coverage: float) -> tuple[float, float]:
    arr = np.asarray(segment, dtype=float)
    left_q = (1.0 - coverage) / 2.0
    right_q = 1.0 - left_q
    return float(np.quantile(arr, left_q)), float(np.quantile(arr, right_q))


def normal_bounds(segment: list[float], coverage: float) -> tuple[float, float]:
    arr = np.asarray(segment, dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    k = NormalDist().inv_cdf((1.0 + coverage) / 2.0)
    return mu - k * sigma, mu + k * sigma


def robust_bounds(segment: list[float], coverage: float) -> tuple[float, float]:
    arr = np.asarray(segment, dtype=float)
    center = float(np.median(arr))
    mad = float(np.median(np.abs(arr - center)))
    sigma = 1.4826 * mad
    k = NormalDist().inv_cdf((1.0 + coverage) / 2.0)
    return center - k * sigma, center + k * sigma


MODEL_FUNCTIONS = {
    "quantile": quantile_bounds,
    "normal": normal_bounds,
    "robust": robust_bounds,
}


def coverage_rate(segment: list[float], lower: float, upper: float) -> float:
    arr = np.asarray(segment, dtype=float)
    if arr.size == 0:
        return 0.0
    inside = np.logical_and(arr >= lower, arr <= upper)
    return float(np.mean(inside))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 1.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def fit_polynomial(times: list[float], values: list[float], max_degree: int = 3) -> dict[str, list[float] | float]:
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)

    if x.size == 0:
        return {
            "degree": 0,
            "coeffs": [0.0],
            "grid_x": [],
            "grid_y": [],
            "r2": 0.0,
        }

    if x.size == 1:
        value = float(y[0])
        return {
            "degree": 0,
            "coeffs": [value],
            "grid_x": [float(x[0])],
            "grid_y": [value],
            "r2": 1.0,
        }

    degree = min(max_degree, int(x.size - 1))
    coeffs = np.polyfit(x, y, degree)
    fitted = np.polyval(coeffs, x)

    grid_x = np.linspace(float(np.min(x)), float(np.max(x)), 120)
    grid_y = np.polyval(coeffs, grid_x)

    return {
        "degree": int(degree),
        "coeffs": [float(item) for item in coeffs],
        "grid_x": [float(item) for item in grid_x],
        "grid_y": [float(item) for item in grid_y],
        "r2": float(r2_score(y, fitted)),
    }


def polynomial_to_text(coeffs: list[float]) -> str:
    degree = len(coeffs) - 1
    parts: list[str] = []

    for idx, coefficient in enumerate(coeffs):
        power = degree - idx
        abs_value = abs(coefficient)

        if power > 1:
            body = f"{abs_value:.4f}·t^{power}"
        elif power == 1:
            body = f"{abs_value:.4f}·t"
        else:
            body = f"{abs_value:.4f}"

        if idx == 0:
            sign = "-" if coefficient < 0 else ""
        else:
            sign = " - " if coefficient < 0 else " + "

        parts.append(f"{sign}{body}")

    return "".join(parts)


def build_window_rows(
    relative_ppm: list[float],
    sample_period_min: float,
    windows: list[int],
    coverage: float,
) -> tuple[list[dict], dict[str, dict[str, list[float]]], dict[str, dict[str, list[float] | list[int]]]]:
    rows: list[dict] = []

    model_series: dict[str, dict[str, list[float]]] = {
        model: {"times": [], "lower": [], "upper": [], "center": []} for model in MODEL_ORDER
    }

    histograms: dict[str, dict[str, list[float] | list[int]]] = {}

    for window in windows:
        points_per_segment = max(1, int(round(window / sample_period_min)))
        segments = split_segments(relative_ppm, points_per_segment)
        if not segments:
            continue

        first_segment = np.asarray(segments[0], dtype=float)
        bins_count = min(16, max(5, int(np.sqrt(first_segment.size))))
        counts, edges = np.histogram(first_segment, bins=bins_count)
        histograms[str(window)] = {
            "counts": [int(item) for item in counts],
            "edges": [float(item) for item in edges],
        }

        row = {
            "window": window,
            "points_per_segment": points_per_segment,
            "segments_count": len(segments),
            "models": {},
        }

        for model in MODEL_ORDER:
            model_fn = MODEL_FUNCTIONS[model]
            bounds = [model_fn(segment, coverage) for segment in segments]

            lowers = [item[0] for item in bounds]
            uppers = [item[1] for item in bounds]
            coverages = [coverage_rate(segment, lo, hi) for segment, (lo, hi) in zip(segments, bounds)]
            widths = [hi - lo for lo, hi in bounds]

            lower_avg = float(np.mean(lowers))
            upper_avg = float(np.mean(uppers))

            row["models"][model] = {
                "lower": lower_avg,
                "upper": upper_avg,
                "center": (lower_avg + upper_avg) / 2.0,
                "width": float(np.mean(widths)),
                "in_sample_coverage": float(np.mean(coverages)),
            }

            model_series[model]["times"].append(float(window))
            model_series[model]["lower"].append(lower_avg)
            model_series[model]["upper"].append(upper_avg)
            model_series[model]["center"].append((lower_avg + upper_avg) / 2.0)

        rows.append(row)

    return rows, model_series, histograms


def normalize_metric(values: list[float]) -> list[float]:
    finite_values = [item for item in values if np.isfinite(item)]

    if not finite_values:
        return [1.0 for _ in values]

    min_value = min(finite_values)
    max_value = max(finite_values)

    if max_value == min_value:
        return [0.0 if np.isfinite(item) else 1.0 for item in values]

    out: list[float] = []
    for item in values:
        if not np.isfinite(item):
            out.append(1.0)
        else:
            out.append((item - min_value) / (max_value - min_value))

    return out


def model_competition(
    relative_ppm: list[float],
    sample_period_min: float,
    windows: list[int],
    coverage: float,
) -> list[dict]:
    raw_rows: list[dict] = []

    for model in MODEL_ORDER:
        model_fn = MODEL_FUNCTIONS[model]
        all_coverages: list[float] = []
        all_widths: list[float] = []

        for window in windows:
            points_per_segment = max(1, int(round(window / sample_period_min)))
            segments = split_segments(relative_ppm, points_per_segment)

            if len(segments) < 2:
                continue

            for index in range(1, len(segments)):
                prev_segment = segments[index - 1]
                current_segment = segments[index]
                lower, upper = model_fn(prev_segment, coverage)
                all_coverages.append(coverage_rate(current_segment, lower, upper))
                all_widths.append(upper - lower)

        if all_coverages:
            coverage_error = abs(float(np.mean(all_coverages)) - coverage)
            mean_width = float(np.mean(all_widths))
            stability = float(np.std(all_widths) / mean_width) if mean_width > 0 else 0.0
        else:
            coverage_error = 1.0
            mean_width = float("inf")
            stability = 1.0

        raw_rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "coverage_error": coverage_error,
                "mean_width": mean_width,
                "stability": stability,
                "idea": MODEL_CRITIQUE[model]["idea"],
                "critique": MODEL_CRITIQUE[model]["critique"],
            }
        )

    coverage_norm = normalize_metric([row["coverage_error"] for row in raw_rows])
    width_norm = normalize_metric([row["mean_width"] for row in raw_rows])
    stability_norm = normalize_metric([row["stability"] for row in raw_rows])

    for index, row in enumerate(raw_rows):
        row["score"] = (
            0.55 * coverage_norm[index]
            + 0.30 * width_norm[index]
            + 0.15 * stability_norm[index]
        )

    ranked_rows = sorted(raw_rows, key=lambda item: item["score"])

    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    return ranked_rows


def build_polynomials(model_series: dict[str, dict[str, list[float]]]) -> dict[str, dict]:
    output: dict[str, dict] = {}

    for model in MODEL_ORDER:
        series = model_series[model]
        times = series["times"]

        lower_fit = fit_polynomial(times, series["lower"], max_degree=3)
        upper_fit = fit_polynomial(times, series["upper"], max_degree=3)

        output[model] = {
            "lower": lower_fit,
            "upper": upper_fit,
            "formula_lower": polynomial_to_text(lower_fit["coeffs"]),
            "formula_upper": polynomial_to_text(upper_fit["coeffs"]),
        }

    return output


def build_analysis(
    signal_values: list[float],
    total_duration_min: float = 60.0,
    coverage: float = 0.95,
    windows: list[int] | None = None,
) -> dict:
    values = [float(item) for item in signal_values]
    windows_clean = clean_windows(windows)

    if not values:
        return {
            "raw_values": [],
            "relative_ppm": [],
            "window_rows": [],
            "model_series": {},
            "competition": [],
            "polynomials": {},
            "histograms": {},
            "raw_stats": {},
            "ppm_stats": {},
            "sample_period_min": 0.0,
            "windows": windows_clean,
        }

    sample_period_min = float(total_duration_min) / float(len(values))
    relative_ppm = compute_relative_ppm(values)

    window_rows, model_series, histograms = build_window_rows(
        relative_ppm=relative_ppm,
        sample_period_min=sample_period_min,
        windows=windows_clean,
        coverage=coverage,
    )

    competition = model_competition(
        relative_ppm=relative_ppm,
        sample_period_min=sample_period_min,
        windows=windows_clean,
        coverage=coverage,
    )

    polynomials = build_polynomials(model_series)

    return {
        "raw_values": values,
        "relative_ppm": relative_ppm,
        "window_rows": window_rows,
        "model_series": model_series,
        "competition": competition,
        "polynomials": polynomials,
        "histograms": histograms,
        "raw_stats": compute_basic_stats(values),
        "ppm_stats": compute_basic_stats(relative_ppm),
        "sample_period_min": sample_period_min,
        "windows": windows_clean,
    }
