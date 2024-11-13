from typing import List
import pandas as pd

from fw_sub_bins import (
    sub_2_bins,
    sub_3_bins,
    fw_bins,
    high_w_6mm,
    low_w_6mm,
    shadow_w_6mm,
    w_10mm,
    high_w_12mm,
    low_w_12mm,
    shadow_w_12mm,
    b_1um,
    b_100nm,
    b_5um,
    ss_5mm,
    dfw
)

# TODO: make this an optional class that requires specification from the user? 

# class sub_bin: 
#     sub: int
#     length: float
#     material: str
#     shadowed: bool
#     low_wetted: bool
#     high_wetted: bool
#     dfw: bool

#     def __init__(
#         self,
#         index: int,
#         sub_bins: list,
#         length: float,
#         material: str,
#         shadowed: bool, 
#         low_wetted: bool, 
#         high_wetted: bool,
#         dfw: bool,
#     ):
#         self.index = index
#         self.sub_bins = sub_bins
#         self.length = length
#         self.material = material
#         self.shadowed = shadowed
#         self.low_wetted = low_wetted
#         self.high_wetted = high_wetted
#         self.dfw = dfw

    

class Bin:
    index: int
    sub_bins: list
    length: float
    material: str
    shadowed: bool
    low_wetted: bool
    high_wetted: bool
    dfw: bool

    def __init__(
        self,
        index: int,
        sub_bin: int,
        length: float,
        material: str,
        shadowed: bool, 
        low_wetted: bool, 
        high_wetted: bool,
        dfw: bool,
    ):
        self.index = index
        self.sub_bins = sub_bin
        self.length = length
        self.material = material
        self.shadowed = shadowed
        self.low_wetted = low_wetted
        self.high_wetted = high_wetted
        self.dfw = dfw

    def __init__(self, sub_bin: int): 
        """Initializes a sub_bins object containing several sub bins.

        Args:
            sub_bins (List, optional): The number of sub_bins in a given bin. 
        """

        sub_bins = list(range(1,sub_bin+1))
        self._sub_bins = sub_bins
    
    def read_wetted_data(self, index: int, filename):
        """Reads wetted/shadowed data from csv file for first wall.

        Args:
            filename (str): filename of csv file with wetted FW data
            index (int): bin number

        Returns:
            Slow/Shigh, Stot, f, DFW for bin

        """

        data = pd.read_csv(filename, skiprows=1, names=range(5))
        data = data.to_numpy()
        return data[self.index - 1]

    def compute_wetted_frac(
    self, index: int, Slow: float, Stot: float, Shigh: float, f: float, low_wetted: bool, high_wetted: bool, shadowed: bool,
):
        """Computes fraction of wetted-ness for first wall sub-bins.

        Args:
            index (int): bin number
            Slow (float): surface area of low wetted area.
            Stot (float): total surface area of bin.
            Shigh (float): surface area of high wetted area.
            f (float): fraction of heat values in low wetted area from SMITER csv files.
            low_wet (Boolean): True if solving for low wetted bin.
            high_wet (Boolean): True if solving for high wetted bin.
            shadowed (Boolean): True if solving for shadowed bin.

        Returns:
            frac: fraction of wetted-ness for sub-bin.

        """
        if self.index in sub_3_bins:
            if low_wetted:
                frac = f * Stot / Slow
            elif high_wetted:
                frac = (1 - f) * Stot / Shigh
            elif shadowed:
                frac = 0.0

        elif self.index in sub_2_bins:
            if low_wetted:
                frac = Stot / Slow
            elif shadowed:
                frac = 0.0

        else:  # div blocks
            frac = 1

        return frac

    # TODO: add tests for find_length
    def find_length(self, index:int, shadowed: bool, low_wetted: bool, high_wetted: bool, dfw: bool):
        """Finds length and material of given bin.

        Args:
            index (int): bin number
            shadowed (bool): True if sub_bin is shadowed
            low_wetted (bool): True if sub_bin is low_wetted
            high_wetted (bool): True if sub_bin is high_wetted
            dfw (bool): True if sub_bin is dfw

        Returns:
            length (float): length of given bin or sub bin.
            material (str): material of given bin or sub bin.

        """

        if self.shadowed: 
            section = 'shadowed'
        if self.low_wetted: 
            section = 'low_wetted'
        if self.high_wetted:
            section = 'high_wetted'
        if self.dfw:
            section = 'dfw'

        if self.index in fw_bins:
            if section == "high_wetted":
                if self.index in high_w_6mm:
                    self.sub_bin.length = 6e-3  # m

                elif self.index in w_10mm:
                    self.sub_bin.length = 10e-3

                elif self.index in high_w_12mm:
                    self.sub_bin.length = 12e-3

                self.sub_bin.material = "W"

            elif section == "low_wetted":
                if self.index in low_w_6mm:
                    self.sub_bin.length = 6e-3  # m
                    self.sub_bin.material = "W"

                elif self.index in w_10mm:
                    self.sub_bin.length = 10e-3
                    self.sub_bin.material = "W"

                elif self.index in low_w_12mm:
                    self.sub_bin.length = 12e-3
                    self.sub_bin.material = "W"

                elif self.index in b_100nm:
                    self.sub_bin.length = 100e-9
                    self.sub_bin.material = "B"

            elif section == "dfw": 
                self.sub_bin.length = 5e-3
                self.sub_bin.material = "SS"

            else:
                if self.index in shadow_w_6mm:
                    self.sub_bin.length = 6e-3  # m
                    self.sub_bin.material = "W"

                elif self.index in w_10mm:
                    self.sub_bin.length = 10e-3
                    self.sub_bin.material = "W"

                elif self.index in shadow_w_12mm:
                    self.sub_bin.length = 12e-3
                    self.sub_bin.material = "W"

                elif self.index in b_1um:
                    self.sub_bin.length = 1e-6
                    self.sub_bin.material = "B"

            return self.sub_bin.length, self.sub_bin.material

        else:
            if self.index in high_w_6mm:
                self.length = 6e-3  # m
                self.material = "W"

            elif self.index in b_1um:
                self.length = 1e-6
                self.material = "B"

            elif self.index in b_5um:
                self.length = 5e-6
                self.material = "B"

            return self.length, self.material

    

class BinCollection:
    def __init__(self, bins: List[Bin] = None):
        """Initializes a BinCollection object containing several bins.

        Args:
            bins: The list of bins in the collection. Each bin is a Bin object.
        """
        self._bins = bins if bins is not None else []

    @property
    def bins(self) -> List[Bin]:
        return self._bins

class Reactor:
    first_wall: BinCollection
    divertor: BinCollection