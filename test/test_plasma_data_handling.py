from hisp.plamsa_data_handling.main import read_wetted_data, compute_wetted_frac
from iter_bins import fw_bins_with_2_subbins, sub_3_bins, fw_bins

import festim as F
import numpy as np

import pytest


@pytest.mark.parametrize(
    "monoblock, expected_Slow, expected_Stot, expected_Shigh, expected_f",
    [(13, 0.1167564588888888, 44.61141841014145, 0.14880546, 0.1587858814448446)],
)
def test_read_wetted_data_sub3(
    monoblock, expected_Slow, expected_Shigh, expected_Stot, expected_f
):
    """Tests if reading wetted data csv correctly for sub3 bins."""

    mb_data = read_wetted_data("Wetted_Frac_Bin_Data.csv", monoblock)

    Slow = float(mb_data[0])
    Shigh = float(mb_data[2])
    Stot = float(mb_data[1])
    f = float(mb_data[3])

    assert expected_Slow == Slow
    assert expected_Shigh == Shigh
    assert expected_f == f
    assert expected_Stot == Stot


@pytest.mark.parametrize(
    "monoblock, expected_Slow, expected_Stot",
    [(17, 4.5144701888888887, 51.5844461141654)],
)
def test_read_wetted_data_sub2(monoblock, expected_Slow, expected_Stot):
    """Tests if reading wetted data csv correctly for sub2 bins."""

    mb_data = read_wetted_data("Wetted_Frac_Bin_Data.csv", monoblock)

    Slow = float(mb_data[0])
    Stot = float(mb_data[1])

    assert expected_Slow == Slow
    assert expected_Stot == Stot


@pytest.mark.parametrize(
    "monoblock, expected_Slow, expected_Stot, DFW",
    [(15, 4.54771414, 17.68441885886407, 11.8)],
)
def test_read_wetted_data_dfw(monoblock, expected_Slow, expected_Stot, DFW):
    """Tests if reading wetted data csv correctly for dfw bins."""

    mb_data = read_wetted_data("Wetted_Frac_Bin_Data.csv", monoblock)

    Slow = float(mb_data[0])
    dfw = float(mb_data[2])
    Stot = float(mb_data[1])

    assert expected_Slow == Slow
    assert expected_Stot == Stot
    assert DFW == dfw


@pytest.mark.parametrize(
    "nb_mb, Slow, Stot, Shigh, f",
    [(2, 0.1467486, 16.111714586068415, 4.40418178, 0.011414484855668546)],
)
def test_compute_wetted_frac_sub3(nb_mb, Slow, Stot, Shigh, f):
    """Tests that compute_wetted_frac works correctly for sub3 bin."""

    # low wetted frac
    expected_frac = f * Stot / Slow
    frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, low_wet=True)
    assert expected_frac == frac

    # high wetted frac
    expected_frac = (1 - f) * Stot / Shigh
    frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, high_wet=True)
    assert expected_frac == frac

    # shadowed
    expected_frac = 0.0
    frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, shadowed=True)
    assert expected_frac == frac


@pytest.mark.parametrize(
    "nb_mb, Slow, Stot, Shigh, f",
    [(18, 8.44870841, 54.484114084704106, np.nan, np.nan)],
)
def test_compute_wetted_frac_sub2(nb_mb, Slow, Stot, Shigh, f):
    """Tests that compute_wetted_frac works correctly for sub2 bin."""

    # low wetted frac
    expected_frac = Stot / Slow
    frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, low_wet=True)
    assert expected_frac == frac

    # shadowed
    expected_frac = 0.0
    frac = compute_wetted_frac(nb_mb, Slow, Stot, Shigh, f, shadowed=True)
    assert expected_frac == frac
