import os

# Matplotlib must not try to open a window on CI.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from masserstein import Spectrum
from masserstein.base_spectrum import BaseSpectrum


@pytest.fixture(autouse=True)
def deterministic_randomness():
    """Every test starts from the same seed, so sampling-based tests are reproducible."""
    import random

    random.seed(20240611)
    np.random.seed(20240611)


@pytest.fixture
def ethanol():
    """A small, normalized isotopic envelope (5 peaks around m/z 46)."""
    s = Spectrum("C2H5OH", threshold=0.001)
    s.normalize()
    return s


@pytest.fixture
def glucose():
    """A normalized envelope well separated from ethanol (around m/z 180)."""
    s = Spectrum("C6H12O6", threshold=0.001)
    s.normalize()
    return s


@pytest.fixture
def two_peaks():
    """A minimal BaseSpectrum with two equally intense peaks."""
    return BaseSpectrum(confs=[(1.0, 0.5), (2.0, 0.5)], label="two_peaks")
