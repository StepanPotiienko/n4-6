from django.test import TestCase

from .services import build_analysis, parse_signal_text


class ServiceTests(TestCase):
    def test_parse_signal_text_handles_comma_decimal(self) -> None:
        raw = "1,000000000E+00\n1,500000000E+00"
        parsed = parse_signal_text(raw)
        self.assertEqual(parsed, [1.0, 1.5])

    def test_build_analysis_returns_expected_windows(self) -> None:
        signal = [1.0 + index * 0.000001 for index in range(120)]
        result = build_analysis(signal, total_duration_min=60.0, coverage=0.95, windows=[1, 2, 5])
        self.assertEqual([row["window"] for row in result["window_rows"]], [1, 2, 5])
        self.assertTrue(result["competition"])
