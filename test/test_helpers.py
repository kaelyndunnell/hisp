from hisp.helpers import gaussian_distribution

import numpy as np


def test_gaussian_distr_area():
    """Generates a gaussian distribution and checks if the area under the curve is 1."""
    x = np.linspace(-10, 10, 1000)
    y = gaussian_distribution([x], mean=0, width=1, mod=np)
    area = np.trapz(y, x)
    assert np.isclose(
        area, 1, atol=1e-2
    ), f"Area under Gaussian distribution is {area} instead of 1"
