import numpy as np
import pytest

from make_iter_bins import FW_bins, Div_bins

def test_wetted_fraction():
    """Tests that wetted fraction is correctly computed 
    for FW sub-bins.
    """""
    # first test FW Bin with 3 sub-bins
    fw_bin = FW_bins.get_bin(8)
    for sub_bin in fw_bin.sub_bins:
        if sub_bin.mode == "shadowed":
            shadowed_frac = sub_bin.wetted_frac
        elif sub_bin.mode == "low_wetted":
            low_wet_frac = sub_bin.wetted_frac
        elif sub_bin.mode == "high_wetted":
            high_wet_frac = sub_bin.wetted_frac

    # 0.41066704,18.184414811440848,4.86154584,0.01654854884176685
    expected_shadowed = 0.0
    expected_low_wet = 0.01654854884176685 * 18.184414811440848 / 0.41066704
    expected_high_wet = (1-0.01654854884176685) * 18.184414811440848 / 4.86154584

    assert round(shadowed_frac,6) == round(expected_shadowed,6)
    assert round(low_wet_frac,6) == round(expected_low_wet,6)
    assert round(high_wet_frac,6) == round(expected_high_wet,6)

    # then test FW bin with 2 sub-bins 
    fw_bin = FW_bins.get_bin(17)

    for sub_bin in fw_bin.sub_bins:
        if sub_bin.mode == "shadowed":
            shadowed_frac = sub_bin.wetted_frac
        elif sub_bin.mode == "wetted":
            low_wet_frac = sub_bin.wetted_frac

    expected_shadowed = 0.0
    expected_low_wet = 54.484114084704106/8.44870841

    assert round(shadowed_frac,6) == round(expected_shadowed,6)
    assert round(low_wet_frac,6) == round(expected_low_wet,6)

@pytest.mark.parametrize("nb_bin, inner_flag, outer_flag", [(19, False, True), (50, True, False), (35, False,False)])
def test_inner_or_outer_bin(nb_bin, inner_flag, outer_flag):
    """Tests that div bins are correctly flagged as inner,
    outer, or neither div bins.
    """""
    div_bin = Div_bins.get_bin(nb_bin)
    div_bin.set_inner_and_outer_bins()

    assert div_bin.inner_bin == inner_flag
    assert div_bin.outer_bin == outer_flag

