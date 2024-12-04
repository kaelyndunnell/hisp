from hisp.scenario import Scenario, Pulse
from hisp.plamsa_data_handling import PlasmaDataHandling
from hisp.bin import DivBin

import numpy as np

risp = Pulse(
    pulse_type="RISP",
    nb_pulses=5,
    ramp_up=10,
    steady_state=250,
    ramp_down=10,
    waiting=1530,
    tritium_fraction=0.0,
)


def test_get_RISP_data_t_rel_zero():
    my_plasmadata = PlasmaDataHandling(
        pulse_type_to_data=None,
        path_to_RISP_data="test/test_data/RISP_data",
        path_to_RISP_wall_data="test/test_data/RISP_Wall_data.dat",
        path_to_ROSP_data="test/test_data/ROSP_data",
    )

    my_bin = DivBin()
    my_bin.index = 60
    my_bin.inner_bin = True

    val_t_zero = my_plasmadata.RISP_data(my_bin, t_rel=0)
    val_t_one = my_plasmadata.RISP_data(my_bin, t_rel=1)

    print(val_t_zero)
    print(val_t_one)
    assert np.testing.assert_array_equal(val_t_zero, val_t_one)
