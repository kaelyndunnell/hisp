from hisp.bin import FWBin3Subs, FWBin2Subs, DivBin, BinCollection, Reactor

total_nb_bins = 64
total_fw_bins = 18

# add all subbins to the FW bins
fw_bins = [FWBin3Subs() for _ in range(total_fw_bins)]

for index in [13, 14, 16, 17]:
    fw_bins[index] = FWBin2Subs()

# tag some with DFW

dfw_indices = [9, 13, 14]
for index in dfw_indices:
    fw_bins[index].add_dfw_bin(material="SS", thickness=5e-3)


FW_bins = BinCollection(fw_bins)

div_bins = [DivBin() for _ in range(18, total_nb_bins)]
Div_bins = BinCollection(div_bins)

my_reactor = Reactor(first_wall=FW_bins, divertor=Div_bins)

# ------- FW BINS -------


for bin_index in [0, 1]:
    fw_bin = FW_bins.get_bin(bin_index)
    for subbin in fw_bin.sub_bins:
        subbin.thickness = 6e-3
        subbin.material = "W"

for bin_index in [2, 3, 4]:
    fw_bin = FW_bins.get_bin(bin_index)
    for subbin in fw_bin.sub_bins:
        subbin.thickness = 10e-3
        subbin.material = "W"

for bin_index in [5, 6, 7, 8, 9, 17]:
    fw_bin = FW_bins.get_bin(bin_index)
    for subbin in fw_bin.sub_bins:
        subbin.thickness = 12e-3
        subbin.material = "W"

for bin_index in [10, 11, 12]:
    FW_bins.get_bin(bin_index).high_wetted_subbin.thickness = 12e-3
    FW_bins.get_bin(bin_index).high_wetted_subbin.material = "W"
    FW_bins.get_bin(bin_index).low_wetted_subbin.thickness = 100e-9
    FW_bins.get_bin(bin_index).low_wetted_subbin.material = "B"
    FW_bins.get_bin(bin_index).shadowed_subbin.thickness = 1e-6
    FW_bins.get_bin(bin_index).shadowed_subbin.material = "B"

for bin_index in [13, 14]:
    FW_bins.get_bin(bin_index).wetted_subbin.thickness = 6e-3
    FW_bins.get_bin(bin_index).wetted_subbin.material = "W"
    FW_bins.get_bin(bin_index).shadowed_subbin.thickness = 1e-6
    FW_bins.get_bin(bin_index).shadowed_subbin.material = "B"

for bin_index in [15]:
    FW_bins.get_bin(bin_index).high_wetted_subbin.thickness = 6e-3
    FW_bins.get_bin(bin_index).high_wetted_subbin.material = "W"
    FW_bins.get_bin(bin_index).low_wetted_subbin.thickness = 100e-9
    FW_bins.get_bin(bin_index).low_wetted_subbin.material = "B"
    FW_bins.get_bin(bin_index).shadowed_subbin.thickness = 1e-6
    FW_bins.get_bin(bin_index).shadowed_subbin.material = "B"

for bin_index in [16]:
    FW_bins.get_bin(bin_index).wetted_subbin.thickness = 6e-3
    FW_bins.get_bin(bin_index).wetted_subbin.material = "W"
    FW_bins.get_bin(bin_index).shadowed_subbin.thickness = 1e-6
    FW_bins.get_bin(bin_index).shadowed_subbin.material = "B"

# test that all bins have a thickness, material and mode
for bin in FW_bins.bins:
    for subbin_index, subbin in enumerate(bin.sub_bins):
        # print(bin.index, subbin_index, subbin.thickness, subbin.material, subbin.mode)
        assert subbin.thickness, f"bin {bin.index} subbin {subbin.thickness}"
        assert subbin.material, f"bin {bin.index} subbin {subbin.material}"
        assert subbin.mode is not None, f"bin {bin.index} subbin {subbin.mode}"


# ------- DIV BINS -------

for bin_index in [18, 19, 20, 21, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]:
    div_bin = Div_bins.get_bin(bin_index)
    div_bin.thickness = 1e-6
    div_bin.material = "B"
    div_bin.set_inner_and_outer_bins()

for bin_index in [
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
]:
    div_bin = Div_bins.get_bin(bin_index)
    div_bin.thickness = 6e-3
    div_bin.material = "W"
    div_bin.set_inner_and_outer_bins()

for bin_index in [42, 43, 44, 45, 57, 58, 59, 60, 61, 62, 63]:
    div_bin = Div_bins.get_bin(bin_index)
    div_bin.thickness = 5e-6
    div_bin.material = "B"
    div_bin.set_inner_and_outer_bins()

for bin in Div_bins.bins:
    # print(bin.index, bin.thickness, bin.material, bin.mode)
    assert bin.thickness is not None, f"bin {bin.index} thickness {bin.thickness}"
    assert bin.material is not None, f"bin {bin.index} material {bin.material}"
    assert bin.mode is not None, f"bin {bin.index} mode {bin.mode}"

# read wetted data

filename = "Wetted_Frac_Bin_Data.csv"
my_reactor.read_wetted_data(filename)


# add start and end points to bins
import pandas as pd

data = pd.read_csv("bin_data.dat", sep=",")


for bin in my_reactor.first_wall.bins + my_reactor.divertor.bins:
    bin.start_point = (data.loc[bin.index]["R_Coord"], data.loc[bin.index]["Z_Coord"])

# end point is the start point of next bin
for bin in my_reactor.first_wall.bins + my_reactor.divertor.bins:
    try:
        next_bin = my_reactor.get_bin(bin.index + 1)
    except ValueError:
        next_bin = my_reactor.get_bin(0)
    bin.end_point = next_bin.start_point

# test
assert len(data) == len(FW_bins.bins) + len(
    Div_bins.bins
), f"{len(data)} {len(FW_bins.bins) + len(Div_bins.bins)}"


for bin in my_reactor.first_wall.bins + my_reactor.divertor.bins:
    assert bin.start_point is not None
    assert bin.end_point is not None

for bin in FW_bins.bins:
    for subbin_index, subbin in enumerate(bin.sub_bins):
        assert (
            subbin.wetted_frac is not None
        ), f"bin {bin.index} subbin {subbin_index} wetted fraction {subbin.wetted_frac}"
        assert subbin.f is not None
        assert subbin.low_wetted_area is not None
