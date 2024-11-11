from hisp.plamsa_data_handling.main import read_wetted_data

import festim as F

import pytest


@pytest.mark.parametrize(
    "monoblock, expected_Slow, expected_Stot, expected_Shigh, expected_f",
    [(14, 0.1167564588888888, 44.61141841014145, 0.14880546, 0.1587858814448446)],
)
def test_read_wetted_data_sub3(
    monoblock, expected_Slow, expected_Shigh, expected_Stot, expected_f
):
    """Tests if reading wetted data csv correctly."""

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
    [(17, 4.5144704, 51.584446114165)],
)
def test_read_wetted_data_sub2(monoblock, expected_Slow, expected_Stot):
    """Tests if reading wetted data csv correctly."""

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
    """Tests if reading wetted data csv correctly."""

    mb_data = read_wetted_data("Wetted_Frac_Bin_Data.csv", monoblock)

    Slow = float(mb_data[0])
    dfw = float(mb_data[2])
    Stot = float(mb_data[1])

    assert expected_Slow == Slow
    assert expected_Stot == Stot
    assert DFW == dfw
