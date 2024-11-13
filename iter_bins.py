# TODO: make this generic

fw_bins = list(range(1, 19))

fw_bins_with_2_subbins = [14, 15, 17, 18]
sub_3_bins = list(range(1, 14)) + [16]
dfw = [10, 14, 15]

high_w_6mm = [
    1,
    2,
    14,
    15,
    16,
    17,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
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
    57,
]
low_w_6mm = [1, 2]
shadow_w_6mm = [1, 2]

w_10mm = [3, 4, 5]  # only blocks with 10mm for all sections

high_w_12mm = [6, 7, 8, 9, 10, 11, 12, 13]
low_w_12mm = [6, 7, 8, 9, 10, 18]
shadow_w_12mm = [6, 7, 8, 9, 10, 18]

b_1um = [
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    33,
    34,
]  # always shadowed for fw bins
b_5um = [58, 59, 60, 61, 62, 63, 64, 43, 44, 45, 46]
b_100nm = [11, 12, 13, 16]  # all low-wetted

ss_5mm = [10, 14, 15]  # always shadowed
