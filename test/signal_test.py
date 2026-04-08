"""Signal stats analysis testing."""

import sys
from pathlib import Path


# Add the root directory to the system path
# To allow importing from other directories
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis import signal_analysis


arr: list[float] = [1, 2, 3, 4, 5]
signal = signal_analysis.SignalStats(arr)


def test_signal_analysis():
    """Test the signal analysis mean function."""
    assert signal.mean == 3


def test_signal_analysis_count():
    """Test if count function returns correct count."""
    assert signal.count == 5


def test_signal_analysis_min():
    """Test if min function returns correct minimum value."""
    assert signal.minimum == 1
