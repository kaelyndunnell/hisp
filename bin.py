from typing import List
import pandas as pd


class SubBin:
    thickness: float
    material: str
    mode: str
    dfw: bool
    parent_bin_index: int

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

    @property
    def shadowed(self) -> bool:
        return self.mode == "shadowed" or self.dfw

    def compute_wetted_frac(self) -> float:
        pass


class FWBin:
    index: int
    sub_bins: List[SubBin]

    def __init__(self, sub_bins: List[SubBin] = None):
        self.sub_bins = sub_bins or []
        self.index = None

    @property
    def shadowed_subbin(self) -> SubBin:
        for subbin in self.sub_bins:
            if subbin.shadowed:
                return subbin

        raise ValueError(f"No shadowed subbin found in bin {self.index}")

    def read_wetted_data(self, filename: str):
        return
        data = pd.read_csv(filename, skiprows=1, names=range(5))
        data = data.to_numpy()
        return data[self.index - 1]

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

    def __init__(self):
        self.index = None
        self.thickness = None
        self.material = None


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
