"""Аналізатор сигналу калібратора Н4-6"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, cast
import csv

DEFAULT_WINDOWS: list[float] = [0.5, 1, 2, 5, 10, 30, 60]
DEFAULT_CONFIDENCE: float = 0.95

_TABLE1_CONFIG: list[tuple[str, str]] = [
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


class SignalParser:
    """
    Завантажує та розбирає сирий текстовий сигнал у список чисел float.

    Підтримувані формати числа: «1,234567E+00» або «1.234567E+00».
    """

    _PATTERN: re.Pattern[str] = re.compile(r"[-+]?\d+[,.]\d+E[-+]?\d+", re.IGNORECASE)
    _ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1251", "latin-1")

    # ── класові методи – кожен є «фабричним конструктором» ──────────── #

    @classmethod
    def parse_text(cls, raw_text: str) -> list[float]:
        """Розбирає рядок і повертає список float-значень."""
        tokens = cls._PATTERN.findall(raw_text)
        return [float(t.replace(",", ".")) for t in tokens]

    @classmethod
    def from_file(cls, path: str | Path) -> list[float]:
        """Читає файл (UTF-8) і повертає розібрані значення."""
        raw_text = Path(path).read_text(encoding="utf-8")
        return cls.parse_text(raw_text)

    @classmethod
    def from_bytes(cls, file_bytes: bytes) -> list[float]:
        """Пробує кілька кодувань і повертає перший успішний розбір."""
        for encoding in cls._ENCODINGS:
            try:
                text = file_bytes.decode(encoding)
                values = cls.parse_text(text)
                if values:
                    return values
            except (UnicodeDecodeError, ValueError):
                continue
        return []

    @classmethod
    def load_default(cls) -> list[float]:
        """Завантажує вбудований файл raw_signal.txt."""
        default_path = Path(__file__).resolve().parents[1] / "data" / "raw_signal.txt"
        return cls.from_file(default_path)


class SignalStats:
    """
    Обчислює і зберігає статистичні характеристики набору значень.

    Усі«важкі» обчислення виконуються ліниво і кешуються через
    property-методи.
    """

    def __init__(self, values: list[float]) -> None:
        self._values: list[float] = [float(v) for v in values]
        self._sorted: list[float] | None = None

    @staticmethod
    def _quick_sort(data: list[float]) -> list[float]:
        if len(data) <= 1:
            return data.copy()
        pivot = data[len(data) // 2]
        less: list[float] = []
        equal: list[float] = []
        greater: list[float] = []
        for v in data:
            if v < pivot:
                less.append(v)
            elif v > pivot:
                greater.append(v)
            else:
                equal.append(v)
        return SignalStats._quick_sort(less) + equal + SignalStats._quick_sort(greater)

    @property
    def _sorted_values(self) -> list[float]:
        if self._sorted is None:
            self._sorted = self._quick_sort(self._values)
        return self._sorted

    @property
    def count(self) -> int:
        """Повертає кількість відліків у наборі"""
        return len(self._values)

    @property
    def mean(self) -> float:
        """Повертає середнє значення або 0.0 для порожнього набору"""
        n = self.count
        if n == 0:
            return 0.0
        total = 0.0
        for v in self._values:
            total += v
        return total / float(n)

    @property
    def median(self) -> float:
        """Повертає медіану або 0.0 для порожнього набору"""
        sv = self._sorted_values
        n = len(sv)
        if n == 0:
            return 0.0
        mid = n // 2
        return sv[mid] if n % 2 == 1 else (sv[mid - 1] + sv[mid]) / 2.0

    @property
    def mode(self) -> float:
        """Повертає моду або 0.0 для порожнього набору"""
        if not self._values:
            return 0.0
        frequencies: dict[float, int] = {}
        for v in self._values:
            frequencies[v] = frequencies.get(v, 0) + 1
        mode_val = self._values[0]
        max_cnt = 0
        for v, cnt in frequencies.items():
            if cnt > max_cnt or (cnt == max_cnt and v < mode_val):
                max_cnt = cnt
                mode_val = v
        return mode_val

    @property
    def minimum(self) -> float:
        """Повертає мінімум або 0.0 для порожнього набору"""
        sv = self._sorted_values
        return sv[0] if sv else 0.0

    @property
    def maximum(self) -> float:
        """Повертає максимум або 0.0 для порожнього набору"""
        sv = self._sorted_values
        return sv[-1] if sv else 0.0

    @property
    def data_range(self) -> float:
        """Повертає розмах (макс - мін) або 0.0 для порожнього набору"""
        return self.maximum - self.minimum

    @property
    def total(self) -> float:
        """Повертає суму всіх відліків або 0.0 для порожнього набору"""
        acc = 0.0
        for v in self._values:
            acc += v
        return acc

    @property
    def variance(self) -> float:
        """Повертає дисперсію вибірки або 0.0 для набору з менше ніж 2 відліками"""
        n = self.count
        if n < 2:
            return 0.0
        m = self.mean
        acc = 0.0
        for v in self._values:
            d = v - m
            acc += d * d
        return acc / float(n - 1)

    @property
    def std(self) -> float:
        """Повертає стандартне відхилення або 0.0 для набору з менше ніж 2 відліками"""
        var = self.variance
        return math.sqrt(var) if var > 0.0 else 0.0

    @property
    def sem(self) -> float:
        """Повертає стандартну помилку або 0.0 для набору з менше ніж 2 відліками"""
        n = self.count
        return self.std / math.sqrt(float(n)) if n > 0 else 0.0

    @property
    def skewness(self) -> float:
        """Повертає асиметричність або 0.0 для набору з менше ніж 3 відліками або нульовим std"""
        n = self.count
        s = self.std
        if n < 3 or s == 0.0:
            return 0.0
        m = self.mean
        acc = sum(((v - m) / s) ** 3 for v in self._values)
        return (float(n) / float((n - 1) * (n - 2))) * acc

    @property
    def excess(self) -> float:
        """Повертає ексцес або 0.0 для набору з менше ніж 4 відліками або нульовим std"""
        n = self.count
        s = self.std
        if n < 4 or s == 0.0:
            return 0.0
        m = self.mean
        acc = sum(((v - m) / s) ** 4 for v in self._values)
        numerator = float(n * (n + 1))
        denominator = float((n - 1) * (n - 2) * (n - 3))
        correction = float(3 * (n - 1) ** 2) / float((n - 2) * (n - 3))
        return (numerator / denominator) * acc - correction

    def to_dict(self) -> dict[str, float]:
        """Повертає всі статистичні характеристики у вигляді словника"""
        return {
            "count": float(self.count),
            "mean": self.mean,
            "sem": self.sem,
            "median": self.median,
            "mode": self.mode,
            "std": self.std,
            "variance": self.variance,
            "excess": self.excess,
            "skewness": self.skewness,
            "range": self.data_range,
            "min": self.minimum,
            "max": self.maximum,
            "sum": self.total,
        }


class HistogramData:
    """
    Будує гістограму з цілочисельними інтервалами шириною 1.

    Нормована колонка HN = count_i / max_count → вісь Y в [0, 1].
    """

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.counts: list[int] = []
        self.edges: list[float] = []
        self.relative: list[float] = []
        self.normalized: list[float] = []
        self.bins_count: int = 0
        self._build()

    def _build(self) -> None:
        n = len(self._values)
        if n == 0:
            return

        sv = sorted(self._values)
        min_val = sv[0]
        max_val = sv[-1]

        bin_start = int(math.floor(min_val))
        bin_end = int(math.ceil(max_val))
        if bin_end <= bin_start:
            bin_end = bin_start + 1

        actual_bins = bin_end - bin_start
        self.edges = [float(bin_start + i) for i in range(actual_bins + 1)]
        self.counts = [0] * actual_bins

        for v in self._values:
            idx = int(math.floor(v)) - bin_start
            idx = max(0, min(idx, actual_bins - 1))
            self.counts[idx] += 1

        total = float(n)
        self.relative = [c / total for c in self.counts]

        max_count = max(self.counts) if self.counts else 0
        self.normalized = (
            [c / float(max_count) for c in self.counts]
            if max_count > 0
            else [0.0] * actual_bins
        )
        self.bins_count = actual_bins

    def quantile(self, q: float) -> float:
        """Повертає квантиль q (0.0 ≤ q ≤ 1.0) або 0.0 для порожньої гістограми"""
        if not self.counts or not self.edges:
            return 0.0
        if q <= 0.0:
            return self.edges[0]
        if q >= 1.0:
            return self.edges[-1]

        total = sum(self.counts)
        if total == 0:
            return 0.0

        cumulative = 0.0
        for idx, cnt in enumerate(self.counts):
            prev = cumulative
            cumulative += cnt / float(total)
            if cumulative >= q:
                left = self.edges[idx]
                right = self.edges[idx + 1]
                if cumulative == prev:
                    return left
                local = (q - prev) / (cumulative - prev)
                return left + (right - left) * local

        return self.edges[-1]

    def coverage_bounds(self, coverage: float) -> tuple[float, float]:
        """Повертає (lower, upper) межі для заданого рівня довіри"""
        alpha = (1.0 - coverage) / 2.0
        return self.quantile(alpha), self.quantile(1.0 - alpha)

    def crossings(self, level: float = 0.1) -> dict[str, float | None]:
        """Повертає перший і останній X, де нормована частота >= level"""
        first_idx: int | None = None
        last_idx: int | None = None
        for idx, hn in enumerate(self.normalized):
            if hn >= level:
                if first_idx is None:
                    first_idx = idx
                last_idx = idx
        if first_idx is None or last_idx is None:
            return {"first_x": None, "last_x": None}
        return {
            "first_x": self.edges[first_idx],
            "last_x": self.edges[last_idx],
        }

    def to_dict(self) -> dict[str, Any]:
        """Повертає всі дані гістограми у вигляді словника"""
        return {
            "counts": self.counts,
            "relative": self.relative,
            "normalized": self.normalized,
            "edges": self.edges,
            "bins_count": self.bins_count,
        }


class WindowAnalyzer:
    """
    Аналізує відносне відхилення (ppm) у заданому часовому вікні.

    Розбиває ряд на рівні сегменти і для кожного будує гістограму
    та обчислює межі довірчого інтервалу.
    """

    def __init__(
        self,
        ppm_values: list[float],
        sample_period_min: float,
        coverage: float,
    ) -> None:
        self._ppm = ppm_values
        self._period = sample_period_min
        self._coverage = coverage

    @staticmethod
    def split(values: list[float], step: int) -> list[list[float]]:
        """Повертає список сегментів, кожен з яких містить не більше step відліків"""
        count = len(values) // step
        return [values[i * step : (i + 1) * step] for i in range(count)]

    def analyze(self, window_min: float) -> dict[str, Any]:
        """Повертає результат аналізу для одного вікна (в хвилинах)"""
        points = (
            max(1, int(round(window_min / self._period))) if self._period > 0 else 1
        )
        segments = self.split(self._ppm, points)

        segment_bounds: list[dict[str, float]] = []
        segment_histograms: list[dict[str, Any]] = []

        for seg in segments:
            hist = HistogramData(seg)
            lower, upper = hist.coverage_bounds(self._coverage)
            segment_histograms.append(hist.to_dict())
            segment_bounds.append(
                {"lower": lower, "upper": upper, "width": upper - lower}
            )

        def _mean_bound(key: str) -> float:
            if not segment_bounds:
                return 0.0
            return sum(b[key] for b in segment_bounds) / len(segment_bounds)

        return {
            "window": window_min,
            "points_per_segment": points,
            "segments_count": len(segments),
            "segment_bounds": segment_bounds,
            "segment_histograms": segment_histograms,
            "avg_lower": _mean_bound("lower"),
            "avg_upper": _mean_bound("upper"),
        }


class PolynomialFitter:
    """
    Апроксимує набір точок поліномом до 3-го ступеня методом МНК.

    Коефіцієнти знаходяться через розв'язання нормальних рівнянь
    методом Гаусса (без використання numpy).
    """

    def __init__(
        self, times: list[float], values: list[float], max_degree: int = 3
    ) -> None:
        self._times = times
        self._values = values
        self._max_degree = max_degree

    @staticmethod
    def _eval(coeffs_asc: list[float], x: float) -> float:
        total = 0.0
        pw = 1.0
        for c in coeffs_asc:
            total += c * pw
            pw *= x
        return total

    @staticmethod
    def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
        size = len(vector)
        for pivot in range(size):
            max_row = pivot
            max_val = abs(matrix[pivot][pivot])
            for row in range(pivot + 1, size):
                val = abs(matrix[row][pivot])
                if val > max_val:
                    max_val = val
                    max_row = row
            if max_val == 0.0:
                return [0.0] * size
            if max_row != pivot:
                matrix[pivot], matrix[max_row] = matrix[max_row], matrix[pivot]
                vector[pivot], vector[max_row] = vector[max_row], vector[pivot]
            pv = matrix[pivot][pivot]
            for col in range(pivot, size):
                matrix[pivot][col] /= pv
            vector[pivot] /= pv
            for row in range(size):
                if row == pivot:
                    continue
                f = matrix[row][pivot]
                if f == 0.0:
                    continue
                for col in range(pivot, size):
                    matrix[row][col] -= f * matrix[pivot][col]
                vector[row] -= f * vector[pivot]
        return vector

    def fit(self) -> dict[str, Any]:
        """Повертає словник з результатами апроксимації"""
        ts, vs = self._times, self._values
        if not ts:
            return {"degree": 0, "coeffs": [0.0], "grid_x": [], "grid_y": [], "r2": 0.0}
        if len(ts) == 1:
            return {
                "degree": 0,
                "coeffs": [vs[0]],
                "grid_x": [ts[0]],
                "grid_y": [vs[0]],
                "r2": 1.0,
            }

        degree = min(self._max_degree, len(ts) - 1)
        size = degree + 1

        matrix = [
            [sum(t ** (r + c) for t in ts) for c in range(size)] for r in range(size)
        ]
        vector = [sum(v * (t**r) for t, v in zip(ts, vs)) for r in range(size)]

        coeffs_asc = self._solve(matrix, vector)
        fitted = [self._eval(coeffs_asc, t) for t in ts]

        # R²
        mean_y = sum(vs) / len(vs)
        ss_tot = sum((y - mean_y) ** 2 for y in vs)
        ss_res = sum((y - yp) ** 2 for y, yp in zip(vs, fitted))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 1.0

        # Grid for smooth curve
        x_min, x_max = min(ts), max(ts)
        if x_max == x_min:
            grid_x = [x_min]
            grid_y = [self._eval(coeffs_asc, x_min)]
        else:
            steps = 120
            grid_x = [x_min + (x_max - x_min) * i / (steps - 1) for i in range(steps)]
            grid_y = [self._eval(coeffs_asc, x) for x in grid_x]

        coeffs_desc = coeffs_asc[::-1]
        return {
            "degree": degree,
            "coeffs": coeffs_desc,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "r2": r2,
        }

    @staticmethod
    def to_formula(coeffs_desc: list[float]) -> str:
        """Перетворює коефіцієнти у рядк виду «a*t^3 + b*t^2 + …»."""
        degree = len(coeffs_desc) - 1
        parts: list[str] = []
        for idx, coef in enumerate(coeffs_desc):
            power = degree - idx
            abs_val = abs(coef)
            body = (
                f"{abs_val:.4f}*t^{power}"
                if power > 1
                else (f"{abs_val:.4f}*t" if power == 1 else f"{abs_val:.4f}")
            )
            sign = (
                ("-" if coef < 0 else "")
                if idx == 0
                else (" - " if coef < 0 else " + ")
            )
            parts.append(f"{sign}{body}")
        return "".join(parts)


class CalibratorAnalysis:
    """
    Повний аналіз вихідного сигналу калібратора Н4-6.

    Використовує SignalStats, HistogramData, WindowAnalyzer та
    PolynomialFitter для обчислення всіх характеристик, що збігаються
    з форматом services.build_analysis().
    """

    def __init__(
        self,
        signal_values: list[float],
        total_duration_min: float = 60.0,
        coverage: float = DEFAULT_CONFIDENCE,
        windows: list[float] | None = None,
    ) -> None:
        self._signal = [float(v) for v in signal_values]
        self._duration = total_duration_min
        self._coverage = coverage
        self._windows = self.clean_windows(windows)

    @staticmethod
    def clean_windows(windows: list[float] | None) -> list[float]:
        """Повертає відфільтрований і відсортований список вікон"""
        if not windows:
            return DEFAULT_WINDOWS.copy()
        cleaned: list[float] = []
        for w in windows:
            if w > 0 and w not in cleaned:
                cleaned.append(w)
        if not cleaned:
            return DEFAULT_WINDOWS.copy()
        for i in range(1, len(cleaned)):
            cur = cleaned[i]
            j = i - 1
            while j >= 0 and cleaned[j] > cur:
                cleaned[j + 1] = cleaned[j]
                j -= 1
            cleaned[j + 1] = cur
        return cleaned

    def compute_ppm(self) -> list[float]:
        """Повертає список відносних відхилень (ppm) для кожного відліку сигналу"""
        values = self._signal
        if not values:
            return []
        m = SignalStats(values).mean
        if m == 0.0:
            return [0.0] * len(values)
        return [(v - m) / m * 1_000_000.0 for v in values]

    def build_interval_analysis(
        self, ppm: list[float], level: float = 0.1
    ) -> dict[str, Any]:
        """Повертає словник з аналізом інтервалів для всього сигналу та його половин"""

        def _analyse(vals: list[float]) -> dict[str, Any]:
            stats = SignalStats(vals)
            hist = HistogramData(vals)
            return {
                "stats": stats.to_dict(),
                "histogram": hist.to_dict(),
                "crossings": hist.crossings(level),
            }

        seg_60 = _analyse(ppm)
        half = len(ppm) // 2
        seg_30_a = _analyse(ppm[:half])
        seg_30_b = _analyse(ppm[half : 2 * half])

        def _avg_opt(a: float | None, b: float | None) -> float | None:
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return (a + b) / 2.0

        return {
            "seg_60": seg_60,
            "seg_30_a": seg_30_a,
            "seg_30_b": seg_30_b,
            "avg_left": _avg_opt(
                seg_30_a["crossings"]["first_x"], seg_30_b["crossings"]["first_x"]
            ),
            "avg_right": _avg_opt(
                seg_30_a["crossings"]["last_x"], seg_30_b["crossings"]["last_x"]
            ),
            "level": level,
        }

    def build_table3(
        self,
        windows: list[float],
        window_map: dict[float, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Повертає список рядків для таблиці 3, де кожен ряд містить
        хвилину та межі для кожного вікна, якщо вони застосовні"""
        total_min = max(1, int(round(self._duration)))
        rows: list[dict[str, Any]] = []
        for minute in range(1, total_min + 1):
            row_values: dict[float, dict[str, float] | None] = {}
            for w in windows:
                info = window_map.get(w)
                if info is None or w <= 0 or (minute % w) > 1e-7:
                    row_values[w] = None
                    continue
                idx = int(round(minute / w)) - 1
                bounds = info["segment_bounds"]
                row_values[w] = bounds[idx] if 0 <= idx < len(bounds) else None
            rows.append({"minute": minute, "values": row_values})
        return rows

    def run(self) -> dict[str, Any]:
        """Виконує повний аналіз і повертає словник результатів."""
        values = self._signal
        windows = self._windows

        if not values:
            empty = SignalStats([]).to_dict()
            empty_hist = HistogramData([]).to_dict()
            empty_fit: dict[str, Any] = {
                "degree": 0,
                "coeffs": [0.0],
                "grid_x": [],
                "grid_y": [],
                "r2": 0.0,
            }
            return {
                "raw_values": [],
                "relative_ppm": [],
                "raw_stats": empty,
                "ppm_stats": empty,
                "table1_rows": [],
                "sample_period_min": 0.0,
                "windows": windows,
                "window_summaries": [],
                "table3_rows": [],
                "table4_rows": [],
                "histograms": {},
                "global_histogram": empty_hist,
                "polynomial": {
                    "lower": empty_fit,
                    "upper": empty_fit,
                    "formula_lower": "0.0000",
                    "formula_upper": "0.0000",
                },
                "interval_analysis": {},
            }

        ppm = self.compute_ppm()
        sample_period = float(self._duration) / float(len(values))

        analyzer = WindowAnalyzer(ppm, sample_period, self._coverage)
        window_map: dict[float, dict[str, Any]] = {}
        window_summaries: list[dict[str, Any]] = []
        histograms: dict[str, Any] = {}

        for w in windows:
            info = analyzer.analyze(w)
            window_map[w] = info
            window_summaries.append(
                {
                    "window": w,
                    "segments_count": info["segments_count"],
                    "points_per_segment": info["points_per_segment"],
                    "avg_lower": info["avg_lower"],
                    "avg_upper": info["avg_upper"],
                }
            )
            histograms[str(w)] = (
                info["segment_histograms"][0]
                if info["segment_histograms"]
                else HistogramData([]).to_dict()
            )

        table3 = self.build_table3(windows, window_map)
        table4: list[dict[str, float]] = [
            {
                "window": w,
                "lower": window_map[w]["avg_lower"],
                "upper": window_map[w]["avg_upper"],
            }
            for w in windows
            if w in window_map
        ]

        raw_stats = SignalStats(values)
        ppm_stats = SignalStats(ppm)

        times = [row["window"] for row in table4]
        lower_fit = PolynomialFitter(times, [r["lower"] for r in table4]).fit()
        upper_fit = PolynomialFitter(times, [r["upper"] for r in table4]).fit()
        polynomial: dict[str, Any] = {
            "lower": lower_fit,
            "upper": upper_fit,
            "formula_lower": PolynomialFitter.to_formula(lower_fit["coeffs"]),
            "formula_upper": PolynomialFitter.to_formula(upper_fit["coeffs"]),
        }

        raw_d = raw_stats.to_dict()
        ppm_d = ppm_stats.to_dict()
        table1: list[dict[str, Any]] = [
            {"label": label, "raw": raw_d.get(key, 0.0), "ppm": ppm_d.get(key, 0.0)}
            for label, key in _TABLE1_CONFIG
        ]

        return {
            "raw_values": values,
            "relative_ppm": ppm,
            "raw_stats": raw_d,
            "ppm_stats": ppm_d,
            "table1_rows": table1,
            "sample_period_min": sample_period,
            "windows": windows,
            "window_summaries": window_summaries,
            "table3_rows": table3,
            "table4_rows": table4,
            "histograms": histograms,
            "polynomial": polynomial,
            "global_histogram": HistogramData(ppm).to_dict(),
            "interval_analysis": self.build_interval_analysis(ppm),
        }


class ResultExporter:
    """
    Зберігає результати аналізу у файли у вказаній директорії
    """

    def __init__(self, result: dict[str, Any], output_dir: str | Path) -> None:
        self._result = result
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, filename: str = "analysis_result.json") -> Path:
        """Зберігає повний результат у JSON"""
        # Підготовка: видаляємо ключі з нескінченними/NaN значеннями
        serializable = self._make_serializable(self._result)
        out_path = self._dir / filename
        out_path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path

    def export_txt(self, filename: str = "analysis_summary.txt") -> Path:
        """Зберігає текстовий звіт зі статистичними характеристиками"""
        r = self._result
        lines: list[str] = [
            "  Аналіз сигналу калібратора Н4-6",
            f"  Кількість відліків   : {int(r['raw_stats']['count'])}",
            f"  Період дискретизації : {r['sample_period_min']:.4f} хв",
            "  Таблиця 1: Статистичні характеристики",
            f"  {'Показник':<28}{'Сигнал':>14}{'ppm':>14}",
        ]
        for row in r.get("table1_rows", []):
            lines.append(f"  {row['label']:<28}{row['raw']:>14.6f}{row['ppm']:>14.6f}")

        lines += [
            "  Таблиця 4: Осереднена U0.95(t)",
            f"  {'Вікно, хв':<12}{'Нижня межа, ppm':>18}{'Верхня межа, ppm':>18}",
        ]
        for row in r.get("table4_rows", []):
            lines.append(
                f"  {row['window']:<12.1f}{row['lower']:>18.6f}{row['upper']:>18.6f}"
            )

        lines += [
            "  Поліном 3-го порядку",
            f"  U_low(t)  = {r['polynomial']['formula_lower']}",
            f"  U_high(t) = {r['polynomial']['formula_upper']}",
        ]

        out_path = self._dir / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def export_csv(self, filename: str = "analysis_table1.csv") -> Path:
        """Зберігає Таблицю у CSV"""
        out_path = self._dir / filename
        with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Показник", "Сигнал", "ppm"])
            for row in self._result.get("table1_rows", []):
                writer.writerow(
                    [row["label"], f"{row['raw']:.9f}", f"{row['ppm']:.9f}"]
                )
        return out_path

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Рекурсивно видаляє нескінченні та NaN-значення"""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            d = cast(dict[str, Any], obj)
            return {k: ResultExporter._make_serializable(v) for k, v in d.items()}
        if isinstance(obj, list):
            lst = cast(list[Any], obj)
            return [ResultExporter._make_serializable(v) for v in lst]
        return obj
