from hisp.bin import Reactor, BinCollection

import matplotlib.pyplot as plt
import numpy as np


def plot_bins(bins: BinCollection):
    """Plots the bins in the collection.

    Args:
        bins: The collection of bins to plot.
    """
    ls = []
    for bin in bins.bins:
        l = plt.plot(
            [bin.start_point[0], bin.end_point[0]],
            [bin.start_point[1], bin.end_point[1]],
            label=f"Bin {bin.index}",
        )
        ls.append(l)
    return ls
