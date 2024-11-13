from typing import List

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
        sub_bins: list,
        length: float,
        material: str,
        shadowed: bool, 
        low_wetted: bool, 
        high_wetted: bool,
        dfw: bool,
    ):
        self.index = index
        self.sub_bins = sub_bins
        self.length = length
        self.material = material
        self.shadowed = shadowed
        self.low_wetted = low_wetted
        self.high_wetted = high_wetted
        self.dfw = dfw

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