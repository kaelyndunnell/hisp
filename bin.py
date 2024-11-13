class BinCollection:
    pass

class Reactor:
    first_wall: BinCollection
    divertor: BinCollection

class Bin:
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

    