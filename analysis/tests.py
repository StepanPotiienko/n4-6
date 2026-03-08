from django.test import TestCase

from .services import build_analysis, compute_basic_stats, parse_signal_text, quick_sort


class ServiceTests(TestCase):
    def test_parse_signal_text_handles_comma_decimal(self) -> None:
        raw = "1,000000000E+00\n1,500000000E+00"
        parsed = parse_signal_text(raw)
        self.assertEqual(parsed, [1.0, 1.5])

    def test_quick_sort_orders_values(self) -> None:
        values = [3.1, -2.0, 3.1, 0.0, 5.4, 1.2]
        self.assertEqual(quick_sort(values), [-2.0, 0.0, 1.2, 3.1, 3.1, 5.4])

    def test_compute_basic_stats_contains_mode_and_sum(self) -> None:
        stats = compute_basic_stats([2.0, 1.0, 2.0, 4.0])
        self.assertEqual(stats["mode"], 2.0)
        self.assertEqual(stats["sum"], 9.0)
        self.assertEqual(stats["count"], 4.0)

    def test_build_analysis_returns_methodical_tables(self) -> None:
        signal = [1.0 + index * 0.000001 for index in range(120)]
        result = build_analysis(signal, total_duration_min=60.0, coverage=0.95, windows=[1, 2, 5])

        self.assertEqual(result["windows"], [1, 2, 5])
        self.assertEqual(len(result["table3_rows"]), 60)
        self.assertEqual([int(row["window"]) for row in result["table4_rows"]], [1, 2, 5])

        minute_2 = result["table3_rows"][1]
        self.assertIsNotNone(minute_2["values"][1])
        self.assertIsNotNone(minute_2["values"][2])

        self.assertIn("formula_lower", result["polynomial"])
        self.assertIn("formula_upper", result["polynomial"])
        self.assertLessEqual(result["polynomial"]["lower"]["degree"], 3)
