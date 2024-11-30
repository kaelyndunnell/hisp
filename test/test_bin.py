import numpy as np
import pytest
import pandas as pd
from hisp.bin import FWBin3Subs, FWBin2Subs, DivBin, FWBin, BinCollection, Reactor

# create Reactor
fw_bins = [FWBin3Subs() for _ in range(18)]
fw_bins[1] = FWBin2Subs()
div_bins = [DivBin() for _ in range(18, 46)]

FW_bins = BinCollection(fw_bins)
Div_bins = BinCollection(div_bins)

my_reactor = Reactor(first_wall=FW_bins, divertor=Div_bins)

# make bins
for bin_index in list(range(18)):
    fw_bin = FW_bins.get_bin(bin_index)
    for subbin in fw_bin.sub_bins:
        subbin.thickness = 6e-3
        subbin.material = "W"

for bin_index in list(range(18, 46)):
    div_bin = Div_bins.get_bin(bin_index)
    div_bin.thickness = 1e-6
    div_bin.material = "B"
    div_bin.set_inner_and_outer_bins()

filename = "test/wetted_test_data.csv"
my_reactor.read_wetted_data(filename)
df = pd.read_csv(filename)


def test_wetted_fraction():
    """Tests that wetted fraction is correctly computed 
    for FW sub-bins.
    """ ""
    # first test FW Bin with 3 sub-bins
    fw_bin = FW_bins.get_bin(0)
    for sub_bin in fw_bin.sub_bins:
        if sub_bin.mode == "shadowed":
            shadowed_frac = sub_bin.wetted_frac
        elif sub_bin.mode == "low_wetted":
            low_wet_frac = sub_bin.wetted_frac
        elif sub_bin.mode == "high_wetted":
            high_wet_frac = sub_bin.wetted_frac

    Slow = df["Slow"][0]
    Stot = df["Stot"][0]
    Shigh = df["Shigh"][0]
    f = df["f"][0]

    expected_shadowed = 0.0
    expected_low_wet = f * Stot / Slow
    expected_high_wet = (1 - f) * Stot / Shigh

    assert round(shadowed_frac, 6) == round(expected_shadowed, 6)
    assert round(low_wet_frac, 6) == round(expected_low_wet, 6)
    assert round(high_wet_frac, 6) == round(expected_high_wet, 6)

    # then test FW bin with 2 sub-bins
    fw_bin = FW_bins.get_bin(1)

    for sub_bin in fw_bin.sub_bins:
        if sub_bin.mode == "shadowed":
            shadowed_frac = sub_bin.wetted_frac
        elif sub_bin.mode == "wetted":
            low_wet_frac = sub_bin.wetted_frac

    Slow = df["Slow"][1]
    Stot = df["Stot"][1]

    expected_shadowed = 0.0
    expected_low_wet = Stot / Slow

    assert round(shadowed_frac, 6) == round(expected_shadowed, 6)
    assert round(low_wet_frac, 6) == round(expected_low_wet, 6)


@pytest.mark.parametrize(
    "nb_bin, inner_flag, outer_flag",
    [(19, False, True), (45, True, False), (35, False, False)],
)
def test_inner_or_outer_bin(nb_bin, inner_flag, outer_flag):
    """Tests that div bins are correctly flagged as inner,
    outer, or neither div bins.
    """ ""
    div_bin = Div_bins.get_bin(nb_bin)
    div_bin.set_inner_and_outer_bins()

    assert div_bin.inner_bin == inner_flag
    assert div_bin.outer_bin == outer_flag


@pytest.mark.parametrize(
    "start, end",
    [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 0), (1, 1)),
    ],
)
def test_length_divbin(start, end):
    my_bin = DivBin()
    my_bin.start_point = start
    my_bin.end_point = end

    length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    assert np.isclose(my_bin.length, length)


@pytest.mark.parametrize(
    "start, end",
    [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 0), (1, 1)),
    ],
)
def test_length_fwbin(start, end):
    my_bin = FWBin()
    my_bin.start_point = start
    my_bin.end_point = end

    length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    assert np.isclose(my_bin.length, length)
