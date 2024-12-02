from typing import List, Tuple
import pandas as pd
import numpy as np


class SubBin:
    thickness: float
    material: str
    mode: str
    dfw: bool
    parent_bin_index: int
    low_wetted_area: float  # m^2
    high_wetted_area: float  # m^2
    total_area: float  # m^2
    f: float  # fraction of heat values in low_wetted_area from SMITER
    # (f = H_low * low_wetted_area /(H_low * low_wetted_area + H_high * high_wetted_area))

    def __init__(
        self,
        mode: str,
        thickness: float = None,
        material: str = None,
    ):
        self.thickness = thickness
        self.material = material
        self.mode = mode
        self.dfw = False
        self.parent_bin_index = None
        self.low_wetted_area = None
        self.high_wetted_area = None
        self.total_area = None
        self.f = None

    @property
    def shadowed(self) -> bool:
        return self.mode == "shadowed" or self.dfw

    @property
    def wetted_frac(self):
        if self.shadowed:
            return 0.0
        elif self.mode == "wetted":
            return self.total_area / self.low_wetted_area

        elif self.mode == "low_wetted":
            return self.f * self.total_area / self.low_wetted_area

        elif self.mode == "high_wetted":
            return (1 - self.f) * self.total_area / self.high_wetted_area


class FWBin:
    index: int
    sub_bins: List[SubBin]
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]

    def __init__(self, sub_bins: List[SubBin] = None):
        self.sub_bins = sub_bins or []
        self.index = None
        self.start_point = None
        self.end_point = None

    @property
    def shadowed_subbin(self) -> SubBin:
        for subbin in self.sub_bins:
            if subbin.shadowed:
                return subbin

        raise ValueError(f"No shadowed subbin found in bin {self.index}")

    @property
    def length(self) -> float:
        """Calculates the poloidal length of the bin (in m).

        Returns:
            The poloidal length of the bin (in m).
        """
        return (
            (self.end_point[0] - self.start_point[0]) ** 2
            + (self.end_point[1] - self.start_point[1]) ** 2
        ) ** 0.5

    def add_dfw_bin(self, **kwargs):
        dfw_bin = SubBin(mode="shadowed", **kwargs)
        dfw_bin.dfw = True  # TODO do we need this?
        self.sub_bins.append(dfw_bin)


class FWBin3Subs(FWBin):
    def __init__(self):
        subbins = [
            SubBin(mode="shadowed"),
            SubBin(mode="low_wetted"),
            SubBin(mode="high_wetted"),
        ]
        super().__init__(subbins)

    @property
    def low_wetted_subbin(self) -> SubBin:
        return self.sub_bins[1]

    @property
    def high_wetted_subbin(self) -> SubBin:
        return self.sub_bins[2]


class FWBin2Subs(FWBin):
    def __init__(self):
        subbins = [
            SubBin(mode="shadowed"),
            SubBin(mode="wetted"),
        ]
        super().__init__(subbins)

    @property
    def wetted_subbin(self) -> SubBin:
        return self.sub_bins[1]


class DivBin:
    index: int
    thickness: float
    material: str
    mode = "wetted"
    inner_bin = bool
    outer_bin = bool
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]

    def __init__(self):
        self.index = None
        self.thickness = None
        self.material = None
        self.inner_bin = False
        self.outer_bin = False
        self.start_point = None
        self.end_point = None

    def set_inner_and_outer_bins(self) -> bool:
        """Flags if a DivBin is an inner target or outer target bin.

        Returns:
            inner_bin: True if inner bin
            outer_bin: True if outer bin
        """
        inner_swept_bins = list(range(45, 64))
        outer_swept_bins = list(range(18, 33))

        if self.index in inner_swept_bins:
            self.inner_bin = True
        elif self.index in outer_swept_bins:
            self.outer_bin = True

    @property
    def length(self) -> float:
        """Calculates the poloidal length of the bin (in m).

        Returns:
            The poloidal length of the bin (in m).
        """
        return (
            (self.end_point[0] - self.start_point[0]) ** 2
            + (self.end_point[1] - self.start_point[1]) ** 2
        ) ** 0.5


class BinCollection:
    def __init__(self, bins: List[FWBin | DivBin] = None):
        """Initializes a BinCollection object containing several bins.

        Args:
            bins: The list of bins in the collection. Each bin is a Bin object.
        """
        self.bins = bins if bins is not None else []

    def get_bin(self, index: int) -> FWBin | DivBin:
        for bin in self.bins:
            if bin.index == index:
                return bin
        raise ValueError(f"No bin found with index {index}")

    def arc_length(self, middle: bool = False):
        """Returns the cumulative length of all bins in the collection.

        Args:
            middle: If True, computes from the middle of each bin.
                If False, computes from the start of each bin.
        """
        if middle:
            middle_of_bins = []
            cumulative_lengths = [0]
            for bin in self.bins:
                middle_of_bins.append(cumulative_lengths[-1] + bin.length / 2)
                cumulative_lengths.append(cumulative_lengths[-1] + bin.length)
            return middle_of_bins
        else:
            return np.cumsum([bin.length for bin in self.bins])


class Reactor:
    first_wall: BinCollection
    divertor: BinCollection

    def __init__(
        self, first_wall: BinCollection = None, divertor: BinCollection = None
    ):
        self.first_wall = first_wall
        self.divertor = divertor
        all_bins = first_wall.bins + divertor.bins
        for i, bin in enumerate(all_bins):
            bin.index = i

        for i, bin in enumerate(first_wall.bins):
            for subbin in bin.sub_bins:
                subbin.parent_bin_index = i

    def get_bin(self, index: int) -> FWBin | DivBin:
        for bin in self.first_wall.bins + self.divertor.bins:
            if bin.index == index:
                return bin
        raise ValueError(f"No bin found with index {index}")

    def read_wetted_data(self, filename: str):
        data = pd.read_csv(filename)

        for fw_bin in self.first_wall.bins:
            for subbin in fw_bin.sub_bins:
                subbin.low_wetted_area = data.iloc[fw_bin.index]["Slow"]
                subbin.high_wetted_area = data.iloc[fw_bin.index]["Shigh"]
                subbin.total_area = data.iloc[fw_bin.index]["Stot"]
                subbin.f = data.iloc[fw_bin.index]["f"]
