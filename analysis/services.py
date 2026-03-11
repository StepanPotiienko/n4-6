from __future__ import annotations

from typing import Any

from .signal_analysis import (
    CalibratorAnalysis,
    HistogramData,
    PolynomialFitter,
    SignalParser,
    SignalStats,
    WindowAnalyzer,
    DEFAULT_WINDOWS,
    DEFAULT_CONFIDENCE,
)

__all__ = [
    "DEFAULT_WINDOWS",
    "DEFAULT_CONFIDENCE",
    "parse_signal_text",
    "load_default_signal",
    "load_signal_from_upload",
    "quick_sort",
    "clean_windows",
    "compute_basic_stats",
    "compute_relative_ppm",
    "split_segments",
    "build_histogram",
    "find_histogram_crossings",
    "histogram_quantile",
    "bounds_from_histogram",
    "build_interval_analysis",
    "build_table3_rows",
    "build_table4_rows",
    "fit_polynomial",
    "polynomial_to_text",
    "build_polynomial_from_table4",
    "build_table1_rows",
    "build_analysis",
]


def parse_signal_text(raw_text: str) -> list[float]:
    return SignalParser.parse_text(raw_text)


def load_default_signal() -> list[float]:
    return SignalParser.load_default()


def load_signal_from_upload(file_bytes: bytes) -> list[float]:
    """Decode uploaded file bytes and parse the signal values."""
    return SignalParser.from_bytes(file_bytes)


def quick_sort(values: list[float]) -> list[float]:
    return SignalStats.sort(values)


def clean_windows(windows: list[float] | None) -> list[float]:
    return CalibratorAnalysis.clean_windows(windows)


def compute_basic_stats(values: list[float]) -> dict[str, float]:
    return SignalStats(values).to_dict()


def compute_relative_ppm(values: list[float]) -> list[float]:
    if not values:
        return []
    m = SignalStats(values).mean
    if m == 0.0:
        return [0.0] * len(values)
    return [(v - m) / m * 1_000_000.0 for v in values]


def split_segments(values: list[float], points_per_segment: int) -> list[list[float]]:
    if points_per_segment <= 0:
        return []
    return WindowAnalyzer.split(values, points_per_segment)


def build_histogram(
    values: list[float], bins_count: int | None = None
) -> dict[str, Any]:
    """Build a histogram with integer-step bins.  bins_count is ignored (kept for API compat)."""
    return HistogramData(values).to_dict()


def find_histogram_crossings(
    histogram: dict[str, Any], level: float = 0.1
) -> dict[str, Any]:
    h = HistogramData.__new__(HistogramData)
    h.normalized = histogram.get("normalized", [])
    h.edges = histogram.get("edges", [])
    return h.crossings(level)


def histogram_quantile(histogram: dict[str, Any], quantile: float) -> float:
    h = HistogramData.__new__(HistogramData)
    h.counts = histogram.get("counts", [])
    h.edges = histogram.get("edges", [])
    return h.quantile(quantile)


def bounds_from_histogram(
    histogram: dict[str, Any], coverage: float
) -> tuple[float, float]:
    h = HistogramData.__new__(HistogramData)
    h.counts = histogram.get("counts", [])
    h.edges = histogram.get("edges", [])
    return h.coverage_bounds(coverage)


def build_interval_analysis(
    relative_ppm: list[float], level: float = 0.1
) -> dict[str, Any]:
    analysis = CalibratorAnalysis(relative_ppm)
    return analysis.build_interval_analysis(relative_ppm, level)


def build_table3_rows(
    windows: list[float],
    window_map: dict[float, dict[str, Any]],
    total_duration_min: float,
) -> list[dict[str, Any]]:
    analysis = CalibratorAnalysis(
        [], total_duration_min=total_duration_min, windows=windows
    )
    return analysis._build_table3(windows, window_map)


def build_table4_rows(
    windows: list[float], window_map: dict[float, dict[str, Any]]
) -> list[dict[str, float]]:
    return [
        {
            "window": float(w),
            "lower": float(window_map[w]["avg_lower"]),
            "upper": float(window_map[w]["avg_upper"]),
        }
        for w in windows
        if w in window_map
    ]


def fit_polynomial(
    times: list[float], values: list[float], max_degree: int = 3
) -> dict[str, Any]:
    return PolynomialFitter(times, values, max_degree).fit()


def polynomial_to_text(coeffs_desc: list[float]) -> str:
    return PolynomialFitter.to_formula(coeffs_desc)


def build_polynomial_from_table4(table4_rows: list[dict[str, float]]) -> dict[str, Any]:
    if not table4_rows:
        empty_fit = {
            "degree": 0,
            "coeffs": [0.0],
            "grid_x": [],
            "grid_y": [],
            "r2": 0.0,
        }
        return {
            "lower": empty_fit,
            "upper": empty_fit,
            "formula_lower": "0.0000",
            "formula_upper": "0.0000",
        }
    times = [r["window"] for r in table4_rows]
    lower_fit = PolynomialFitter(times, [r["lower"] for r in table4_rows]).fit()
    upper_fit = PolynomialFitter(times, [r["upper"] for r in table4_rows]).fit()
    return {
        "lower": lower_fit,
        "upper": upper_fit,
        "formula_lower": PolynomialFitter.to_formula(lower_fit["coeffs"]),
        "formula_upper": PolynomialFitter.to_formula(upper_fit["coeffs"]),
    }


def build_table1_rows(
    raw_stats: dict[str, float], ppm_stats: dict[str, float]
) -> list[dict[str, Any]]:
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
    return [
        {"label": lbl, "raw": raw_stats.get(k, 0.0), "ppm": ppm_stats.get(k, 0.0)}
        for lbl, k in config
    ]


def build_analysis(
    signal_values: list[float],
    total_duration_min: float = 60.0,
    coverage: float = DEFAULT_CONFIDENCE,
    windows: list[float] | None = None,
) -> dict[str, Any]:
    """Делегує до CalibratorAnalysis.run() — точка входу для views.py."""
    return CalibratorAnalysis(
        signal_values=signal_values,
        total_duration_min=total_duration_min,
        coverage=coverage,
        windows=windows,
    ).run()
