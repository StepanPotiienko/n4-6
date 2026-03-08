from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

DEFAULT_WINDOWS = [0.5, 1, 2, 5, 10, 30, 60]
DEFAULT_CONFIDENCE = 0.95


def parse_signal_text(raw_text: str) -> list[float]:
    tokens = re.findall(r"[-+]?\d+[,.]\d+E[-+]?\d+", raw_text, flags=re.IGNORECASE)
    return [float(token.replace(",", ".")) for token in tokens]


def load_default_signal() -> list[float]:
    file_path = Path(__file__).resolve().parents[1] / "data" / "raw_signal.txt"
    raw_text = file_path.read_text(encoding="utf-8")
    return parse_signal_text(raw_text)


def load_signal_from_upload(file_bytes: bytes) -> list[float]:
    """Decode uploaded file bytes and parse the signal values."""
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            raw_text = file_bytes.decode(encoding)
            values = parse_signal_text(raw_text)
            if values:
                return values
        except (UnicodeDecodeError, ValueError):
            continue
    return []


def quick_sort(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return values.copy()

    pivot = values[len(values) // 2]
    less: list[float] = []
    equal: list[float] = []
    greater: list[float] = []

    for value in values:
        if value < pivot:
            less.append(value)
        elif value > pivot:
            greater.append(value)
        else:
            equal.append(value)

    return quick_sort(less) + equal + quick_sort(greater)


def clean_windows(windows: list[float] | None) -> list[float]:
    if not windows:
        return DEFAULT_WINDOWS.copy()
 
    cleaned: list[float] = []
    for window in windows:
        if window > 0 and window not in cleaned:
            cleaned.append(window)
 
    if not cleaned:
        return DEFAULT_WINDOWS.copy()
 
    # Keep ordering deterministic without using built-in sorting helpers.
    for index in range(1, len(cleaned)):
        current = cleaned[index]
        cursor = index - 1
        while cursor >= 0 and cleaned[cursor] > current:
            cleaned[cursor + 1] = cleaned[cursor]
            cursor -= 1
        cleaned[cursor + 1] = current
 
    return cleaned


def _sum(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _sum(values) / float(len(values))


def _min(values: list[float]) -> float:
    if not values:
        return 0.0
    current = values[0]
    for value in values[1:]:
        if value < current:
            current = value
    return current


def _max(values: list[float]) -> float:
    if not values:
        return 0.0
    current = values[0]
    for value in values[1:]:
        if value > current:
            current = value
    return current


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0

    middle = n // 2
    if n % 2 == 1:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def _mode(values: list[float]) -> float:
    if not values:
        return 0.0

    frequencies: dict[float, int] = {}
    for value in values:
        frequencies[value] = frequencies.get(value, 0) + 1

    mode_value = values[0]
    max_count = 0
    for value, count in frequencies.items():
        if count > max_count:
            max_count = count
            mode_value = value
        elif count == max_count and value < mode_value:
            mode_value = value

    return mode_value


def _sample_variance(values: list[float], mean_value: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0

    acc = 0.0
    for value in values:
        delta = value - mean_value
        acc += delta * delta
    return acc / float(n - 1)


def _sample_std(variance_value: float) -> float:
    if variance_value <= 0.0:
        return 0.0
    return math.sqrt(variance_value)


def _standard_error(std_value: float, n: int) -> float:
    if n == 0:
        return 0.0
    return std_value / math.sqrt(float(n))


def _skewness(values: list[float], mean_value: float, sample_std: float) -> float:
    n = len(values)
    if n < 3 or sample_std == 0.0:
        return 0.0

    acc = 0.0
    for value in values:
        acc += ((value - mean_value) / sample_std) ** 3

    return (float(n) / float((n - 1) * (n - 2))) * acc


def _excess(values: list[float], mean_value: float, sample_std: float) -> float:
    n = len(values)
    if n < 4 or sample_std == 0.0:
        return 0.0

    acc = 0.0
    for value in values:
        acc += ((value - mean_value) / sample_std) ** 4

    numerator = float(n * (n + 1))
    denominator = float((n - 1) * (n - 2) * (n - 3))
    correction = float(3 * (n - 1) * (n - 1)) / float((n - 2) * (n - 3))
    return (numerator / denominator) * acc - correction


def compute_basic_stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "sem": 0.0,
            "median": 0.0,
            "mode": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "excess": 0.0,
            "skewness": 0.0,
            "range": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0,
        }

    sorted_values = quick_sort(values)

    mean_value = _mean(values)
    variance_value = _sample_variance(values, mean_value)
    std_value = _sample_std(variance_value)

    min_value = sorted_values[0]
    max_value = sorted_values[-1]

    return {
        "count": float(n),
        "mean": mean_value,
        "sem": _standard_error(std_value, n),
        "median": _median(sorted_values),
        "mode": _mode(values),
        "std": std_value,
        "variance": variance_value,
        "excess": _excess(values, mean_value, std_value),
        "skewness": _skewness(values, mean_value, std_value),
        "range": max_value - min_value,
        "min": min_value,
        "max": max_value,
        "sum": _sum(values),
    }


def compute_relative_ppm(values: list[float]) -> list[float]:
    if not values:
        return []

    mean_value = _mean(values)
    if mean_value == 0.0:
        return [0.0 for _ in values]

    output: list[float] = []
    for value in values:
        delta = (value - mean_value) / mean_value
        output.append(delta * 1_000_000.0)

    return output


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


def _sturges_bin_count(n: int) -> int:
    if n <= 1:
        return 1
    bins = int(round(1.0 + 3.322 * math.log10(float(n))))
    return max(1, bins)


def build_histogram(values: list[float], bins_count: int | None = None) -> dict[str, Any]:
    """Build a histogram with integer-step bins from floor(min) to ceil(max).

    Each bin has width = 1 (one integer ppm unit).  The ``bins_count`` parameter
    is ignored — it is kept for API compatibility but the bin layout is always
    determined by the range of ``values``.

    Normalized column (HN): count_i / max_count  →  Y axis in [0, 1].
    """
    n = len(values)
    if n == 0:
        return {
            "counts": [],
            "relative": [],
            "normalized": [],
            "edges": [],
            "bins_count": 0,
        }

    sorted_values = quick_sort(values)
    min_value = sorted_values[0]
    max_value = sorted_values[-1]

    # Integer bin edges: floor(min) … ceil(max), step = 1
    bin_start = int(math.floor(min_value))
    bin_end = int(math.ceil(max_value))

    # Guard: at least one bin even when min == max
    if bin_end <= bin_start:
        bin_end = bin_start + 1

    actual_bins = bin_end - bin_start  # number of bins
    edges = [float(bin_start + i) for i in range(actual_bins + 1)]

    counts = [0 for _ in range(actual_bins)]
    for value in values:
        # Place value in the correct integer bin
        idx = int(math.floor(value)) - bin_start
        if idx < 0:
            idx = 0
        if idx >= actual_bins:
            idx = actual_bins - 1
        counts[idx] += 1

    total = float(n)
    relative = [count / total for count in counts]

    max_count = 0
    for count in counts:
        if count > max_count:
            max_count = count

    if max_count == 0:
        normalized = [0.0 for _ in counts]
    else:
        normalized = [count / float(max_count) for count in counts]

    return {
        "counts": counts,
        "relative": relative,
        "normalized": normalized,
        "edges": edges,
        "bins_count": actual_bins,
    }


def find_histogram_crossings(histogram: dict[str, Any], level: float = 0.1) -> dict[str, Any]:
    """Find first and last bin where normalized frequency >= level.

    Crossing is detected bar-by-bar (no interpolation).  The returned X values
    are the left edges of the first and last qualifying bins.

    Returns a dict with keys: ``first_x``, ``last_x`` (both ``None`` if no bin
    reaches the level).
    """
    normalized: list[float] = histogram.get("normalized", [])
    edges: list[float] = histogram.get("edges", [])

    if not normalized or not edges:
        return {"first_x": None, "last_x": None}

    first_idx: int | None = None
    last_idx: int | None = None

    for idx, hn in enumerate(normalized):
        if hn >= level:
            if first_idx is None:
                first_idx = idx
            last_idx = idx

    if first_idx is None:
        return {"first_x": None, "last_x": None}

    return {
        "first_x": edges[first_idx],
        "last_x": edges[last_idx],
    }


def histogram_quantile(histogram: dict[str, Any], quantile: float) -> float:
    counts: list[int] = histogram["counts"]
    edges: list[float] = histogram["edges"]

    if not counts or not edges:
        return 0.0

    if quantile <= 0.0:
        return edges[0]
    if quantile >= 1.0:
        return edges[-1]

    total = 0
    for count in counts:
        total += count

    if total == 0:
        return 0.0

    cumulative = 0.0
    for index, count in enumerate(counts):
        previous = cumulative
        cumulative += count / float(total)

        if cumulative >= quantile:
            left = edges[index]
            right = edges[index + 1]

            if cumulative == previous:
                return left

            local = (quantile - previous) / (cumulative - previous)
            return left + (right - left) * local

    return edges[-1]


def bounds_from_histogram(histogram: dict[str, Any], coverage: float) -> tuple[float, float]:
    alpha = (1.0 - coverage) / 2.0
    lower = histogram_quantile(histogram, alpha)
    upper = histogram_quantile(histogram, 1.0 - alpha)
    return lower, upper


def _mean_bounds(bounds: list[dict[str, float]], key: str) -> float:
    if not bounds:
        return 0.0

    total = 0.0
    for item in bounds:
        total += item[key]
    return total / float(len(bounds))


def _window_points(window: float, sample_period_min: float) -> int:
    if sample_period_min <= 0.0:
        return 0
    return max(1, int(round(float(window) / sample_period_min)))


def _window_analysis(
    relative_ppm: list[float],
    sample_period_min: float,
    window: float,
    coverage: float,
) -> dict[str, Any]:
    points_per_segment = _window_points(window, sample_period_min)
    segments = split_segments(relative_ppm, points_per_segment)

    segment_bounds: list[dict[str, float]] = []
    segment_histograms: list[dict[str, Any]] = []

    for segment in segments:
        histogram = build_histogram(segment)
        lower, upper = bounds_from_histogram(histogram, coverage)

        segment_histograms.append(histogram)
        segment_bounds.append(
            {
                "lower": lower,
                "upper": upper,
                "width": upper - lower,
            }
        )

    return {
        "window": window,
        "points_per_segment": points_per_segment,
        "segments_count": len(segments),
        "segment_bounds": segment_bounds,
        "segment_histograms": segment_histograms,
        "avg_lower": _mean_bounds(segment_bounds, "lower"),
        "avg_upper": _mean_bounds(segment_bounds, "upper"),
    }


def build_interval_analysis(relative_ppm: list[float], level: float = 0.1) -> dict[str, Any]:
    """Compute stats + histogram + crossings for the 60-min segment and two 30-min segments.

    Assumes the full ppm list represents 60 minutes:
    - seg_60: all points → 1 histogram
    - seg_30_a: first half → histogram A
    - seg_30_b: second half → histogram B

    For the 30-min pair, the average crossings are computed:
    - avg_left  = mean(A.first_x, B.first_x)
    - avg_right = mean(A.last_x,  B.last_x)
    """

    def _analyse_segment(values: list[float]) -> dict[str, Any]:
        stats = compute_basic_stats(values)
        histogram = build_histogram(values)
        crossings = find_histogram_crossings(histogram, level)
        return {
            "stats": stats,
            "histogram": histogram,
            "crossings": crossings,
        }

    seg_60 = _analyse_segment(relative_ppm)

    half = len(relative_ppm) // 2
    seg_30_a = _analyse_segment(relative_ppm[:half])
    seg_30_b = _analyse_segment(relative_ppm[half: 2 * half])

    # Average crossings for the two 30-min segments
    def _avg_opt(a: float | None, b: float | None) -> float | None:
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return (a + b) / 2.0

    avg_left = _avg_opt(
        seg_30_a["crossings"]["first_x"],
        seg_30_b["crossings"]["first_x"],
    )
    avg_right = _avg_opt(
        seg_30_a["crossings"]["last_x"],
        seg_30_b["crossings"]["last_x"],
    )

    return {
        "seg_60": seg_60,
        "seg_30_a": seg_30_a,
        "seg_30_b": seg_30_b,
        "avg_left": avg_left,
        "avg_right": avg_right,
        "level": level,
    }


def build_table3_rows(
    windows: list[float],
    window_map: dict[float, dict[str, Any]],
    total_duration_min: float,
) -> list[dict[str, Any]]:
    total_minutes = max(1, int(round(total_duration_min)))
    rows: list[dict[str, Any]] = []

    for minute in range(1, total_minutes + 1):
        row_values: dict[float, dict[str, float] | None] = {}

        for window in windows:
            info = window_map.get(window)
            if info is None:
                row_values[window] = None
                continue

            # For 0.5 window, minute 1, 2, 3... are all valid endpoints.
            # minute % window should be 0 or very close.
            if window <= 0 or (minute % window) > 1e-7:
                row_values[window] = None
                continue

            index = int(round(minute / window)) - 1
            bounds = info["segment_bounds"]
            if 0 <= index < len(bounds):
                row_values[window] = bounds[index]
            else:
                row_values[window] = None

        rows.append({"minute": minute, "values": row_values})

    return rows


def build_table4_rows(windows: list[float], window_map: dict[float, dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    for window in windows:
        info = window_map.get(window)
        if info is None:
            continue

        rows.append(
            {
                "window": float(window),
                "lower": float(info["avg_lower"]),
                "upper": float(info["avg_upper"]),
            }
        )

    return rows


def _poly_eval(coeffs_asc: list[float], x_value: float) -> float:
    total = 0.0
    current_power = 1.0
    for coefficient in coeffs_asc:
        total += coefficient * current_power
        current_power *= x_value
    return total


def _r2_score(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) < 2:
        return 1.0

    mean_value = _mean(y_true)
    ss_total = 0.0
    ss_residual = 0.0

    for true_value, pred_value in zip(y_true, y_pred):
        delta_total = true_value - mean_value
        delta_residual = true_value - pred_value
        ss_total += delta_total * delta_total
        ss_residual += delta_residual * delta_residual

    if ss_total == 0.0:
        return 1.0

    return 1.0 - ss_residual / ss_total


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)

    for pivot in range(size):
        max_row = pivot
        max_value = abs(matrix[pivot][pivot])

        for row in range(pivot + 1, size):
            value = abs(matrix[row][pivot])
            if value > max_value:
                max_value = value
                max_row = row

        if max_value == 0.0:
            return [0.0 for _ in range(size)]

        if max_row != pivot:
            matrix[pivot], matrix[max_row] = matrix[max_row], matrix[pivot]
            vector[pivot], vector[max_row] = vector[max_row], vector[pivot]

        pivot_value = matrix[pivot][pivot]
        for col in range(pivot, size):
            matrix[pivot][col] /= pivot_value
        vector[pivot] /= pivot_value

        for row in range(size):
            if row == pivot:
                continue

            factor = matrix[row][pivot]
            if factor == 0.0:
                continue

            for col in range(pivot, size):
                matrix[row][col] -= factor * matrix[pivot][col]
            vector[row] -= factor * vector[pivot]

    return vector


def fit_polynomial(times: list[float], values: list[float], max_degree: int = 3) -> dict[str, Any]:
    if not times:
        return {
            "degree": 0,
            "coeffs": [0.0],
            "grid_x": [],
            "grid_y": [],
            "r2": 0.0,
        }

    if len(times) == 1:
        return {
            "degree": 0,
            "coeffs": [values[0]],
            "grid_x": [times[0]],
            "grid_y": [values[0]],
            "r2": 1.0,
        }

    degree = min(max_degree, len(times) - 1)
    size = degree + 1

    matrix: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
    vector: list[float] = [0.0 for _ in range(size)]

    for row in range(size):
        for col in range(size):
            power = row + col
            total = 0.0
            for time in times:
                total += time ** power
            matrix[row][col] = total

        rhs = 0.0
        for time, value in zip(times, values):
            rhs += value * (time ** row)
        vector[row] = rhs

    coeffs_asc = _solve_linear_system(matrix, vector)
    fitted = [_poly_eval(coeffs_asc, time) for time in times]

    x_min = _min(times)
    x_max = _max(times)
    grid_x: list[float] = []
    grid_y: list[float] = []

    if x_max == x_min:
        grid_x = [x_min]
        grid_y = [_poly_eval(coeffs_asc, x_min)]
    else:
        steps = 120
        for index in range(steps):
            ratio = float(index) / float(steps - 1)
            x_value = x_min + ratio * (x_max - x_min)
            grid_x.append(x_value)
            grid_y.append(_poly_eval(coeffs_asc, x_value))

    coeffs_desc = [coeffs_asc[index] for index in range(len(coeffs_asc) - 1, -1, -1)]

    return {
        "degree": degree,
        "coeffs": coeffs_desc,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "r2": _r2_score(values, fitted),
    }


def polynomial_to_text(coeffs_desc: list[float]) -> str:
    degree = len(coeffs_desc) - 1
    parts: list[str] = []

    for index, coefficient in enumerate(coeffs_desc):
        power = degree - index
        abs_value = abs(coefficient)

        if power > 1:
            body = f"{abs_value:.4f}*t^{power}"
        elif power == 1:
            body = f"{abs_value:.4f}*t"
        else:
            body = f"{abs_value:.4f}"

        if index == 0:
            sign = "-" if coefficient < 0 else ""
        else:
            sign = " - " if coefficient < 0 else " + "

        parts.append(f"{sign}{body}")

    return "".join(parts)


def build_polynomial_from_table4(table4_rows: list[dict[str, float]]) -> dict[str, Any]:
    if not table4_rows:
        empty_fit = {"degree": 0, "coeffs": [0.0], "grid_x": [], "grid_y": [], "r2": 0.0}
        return {
            "lower": empty_fit,
            "upper": empty_fit,
            "formula_lower": "0.0000",
            "formula_upper": "0.0000",
        }

    times = [row["window"] for row in table4_rows]
    lower_values = [row["lower"] for row in table4_rows]
    upper_values = [row["upper"] for row in table4_rows]

    lower_fit = fit_polynomial(times, lower_values, max_degree=3)
    upper_fit = fit_polynomial(times, upper_values, max_degree=3)

    return {
        "lower": lower_fit,
        "upper": upper_fit,
        "formula_lower": polynomial_to_text(lower_fit["coeffs"]),
        "formula_upper": polynomial_to_text(upper_fit["coeffs"]),
    }


def build_table1_rows(raw_stats: dict[str, float], ppm_stats: dict[str, float]) -> list[dict[str, Any]]:
    config = [
        ("Середнє", "mean"),
        ("Стандартна помилка", "sem"),
        ("Медіана", "median"),
        ("Мода", "mode"),
        ("Стандартне відхилення", "std"),
        ("Дисперсія вибірки", "variance"),
        ("Ексцес", "excess"),
        ("Асиметричність", "skewness"),
        ("Інтервал", "range"),
        ("Мінімум", "min"),
        ("Максимум", "max"),
        ("Сума", "sum"),
        ("Відлік", "count"),
    ]

    rows: list[dict[str, Any]] = []
    for label, key in config:
        rows.append(
            {
                "label": label,
                "raw": raw_stats.get(key, 0.0),
                "ppm": ppm_stats.get(key, 0.0),
            }
        )

    return rows


def build_analysis(
    signal_values: list[float],
    total_duration_min: float = 60.0,
    coverage: float = DEFAULT_CONFIDENCE,
    windows: list[float] | None = None,
) -> dict[str, Any]:
    values = [float(item) for item in signal_values]
    windows_clean = clean_windows(windows)

    if not values:
        empty_stats = compute_basic_stats([])
        return {
            "raw_values": [],
            "relative_ppm": [],
            "raw_stats": empty_stats,
            "ppm_stats": empty_stats,
            "table1_rows": [],
            "sample_period_min": 0.0,
            "windows": windows_clean,
            "window_summaries": [],
            "table3_rows": [],
            "table4_rows": [],
            "histograms": {},
            "polynomial": {
                "lower": {"degree": 0, "coeffs": [0.0], "grid_x": [], "grid_y": [], "r2": 0.0},
                "upper": {"degree": 0, "coeffs": [0.0], "grid_x": [], "grid_y": [], "r2": 0.0},
                "formula_lower": "0.0000",
                "formula_upper": "0.0000",
            },
            "global_histogram": {
                "counts": [],
                "relative": [],
                "normalized": [],
                "edges": [],
                "bins_count": 0,
            },
        }

    sample_period_min = float(total_duration_min) / float(len(values))
    relative_ppm = compute_relative_ppm(values)

    window_map: dict[float, dict[str, Any]] = {}
    window_summaries: list[dict[str, Any]] = []
    histograms: dict[str, Any] = {}

    for window in windows_clean:
        info = _window_analysis(relative_ppm, sample_period_min, window, coverage)
        window_map[window] = info
        window_summaries.append(
            {
                "window": window,
                "segments_count": info["segments_count"],
                "points_per_segment": info["points_per_segment"],
                "avg_lower": info["avg_lower"],
                "avg_upper": info["avg_upper"],
            }
        )

        if info["segment_histograms"]:
            histograms[str(window)] = info["segment_histograms"][0]
        else:
            histograms[str(window)] = {
                "counts": [],
                "relative": [],
                "normalized": [],
                "edges": [],
                "bins_count": 0,
            }

    table3_rows = build_table3_rows(windows_clean, window_map, total_duration_min)
    table4_rows = build_table4_rows(windows_clean, window_map)

    raw_stats = compute_basic_stats(values)
    ppm_stats = compute_basic_stats(relative_ppm)

    polynomial = build_polynomial_from_table4(table4_rows)

    interval_analysis = build_interval_analysis(relative_ppm)

    return {
        "raw_values": values,
        "relative_ppm": relative_ppm,
        "raw_stats": raw_stats,
        "ppm_stats": ppm_stats,
        "table1_rows": build_table1_rows(raw_stats, ppm_stats),
        "sample_period_min": sample_period_min,
        "windows": windows_clean,
        "window_summaries": window_summaries,
        "table3_rows": table3_rows,
        "table4_rows": table4_rows,
        "histograms": histograms,
        "polynomial": polynomial,
        "global_histogram": build_histogram(relative_ppm),
        "interval_analysis": interval_analysis,
    }
