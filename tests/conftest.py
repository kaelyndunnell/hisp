"""
Shared test fixtures for the HISP test suite.

Defines lightweight fake dataclasses that mimic the PFC-TT interface
without importing from it.
"""
import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import List


# ===========================================================================
# Fake dataclasses mimicking PFC-TT interfaces
# ===========================================================================

@dataclass
class FakeTrap:
    """Mimics Material trap parameters from PFC-TT."""
    Trap_density: float = 1e-3  # atomic fraction
    k_0: float = 8.96e-17
    E_k: float = 0.2
    p_0: float = 1e13
    E_p: float = 0.87


@dataclass
class FakeMaterial:
    """Mimics the Material class from PFC-TT."""
    name: str = "W"
    D0: float = 4.1e-7  # m^2/s
    E_D: float = 0.39   # eV
    Mat_density: float = 6.3e28  # atoms/m^3
    traps: List[FakeTrap] = field(default_factory=lambda: [FakeTrap()])
    K_R: float = 7.94e-17
    E_R: float = -2.0
    k_d0: float = 0.0
    E_kd: float = 0.0

    @property
    def N_traps(self) -> int:
        return len(self.traps)


@dataclass
class FakeBinConfig:
    """Mimics BinConfiguration from PFC-TT."""
    bc_plasma_facing_surface: str = "Dirichlet - 0 concentration + Implantation"
    bc_rear_surface: str = "Dirichlet - 0 concentration"
    atol: float = 1e10
    rtol: float = 1e-10


@dataclass
class FakeBin:
    """Mimics the Bin class from PFC-TT."""
    bin_number: int = 1
    sim_id: int = 0
    flux_id: int = 0
    thickness: float = 6e-3  # 6 mm
    copper_thickness: float = 1e-3  # 1 mm
    material: FakeMaterial = field(default_factory=FakeMaterial)
    bin_configuration: FakeBinConfig = field(default_factory=FakeBinConfig)
    implantation_params: dict = field(default_factory=lambda: {
        'ion': {'implantation_range': 3e-9, 'width': 1e-9, 'reflection_coefficient': 0.0},
        'atom': {'implantation_range': 3e-9, 'width': 1e-9, 'reflection_coefficient': 0.0},
    })


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def fake_material():
    """A single-trap tungsten material."""
    return FakeMaterial()


@pytest.fixture
def fake_material_2traps():
    """A two-trap tungsten material."""
    return FakeMaterial(
        traps=[
            FakeTrap(Trap_density=1e-3, E_p=0.87),
            FakeTrap(Trap_density=5e-4, E_p=1.50),
        ]
    )


@pytest.fixture
def fake_bin_config():
    return FakeBinConfig()


@pytest.fixture
def fake_bin():
    """Default FakeBin with tungsten material."""
    return FakeBin()


@pytest.fixture
def fake_bin_B():
    """FakeBin variant for boron (thin layer)."""
    return FakeBin(
        bin_number=2,
        thickness=2e-6,  # 2 µm typical boron layer
        copper_thickness=None,
        material=FakeMaterial(
            name="B",
            D0=1e-10,
            E_D=0.5,
            Mat_density=1.3e29,
            traps=[FakeTrap(Trap_density=2e-3, E_p=1.0)],
            K_R=1e-16,
            E_R=-1.5,
        ),
    )


@pytest.fixture
def T_fn():
    """Simple constant temperature function T(x, t) = 500 K."""
    def _T(x, t):
        return np.full_like(x[0], 500.0)
    return _T


@pytest.fixture
def flux_fn():
    """Constant particle flux function: 1e20 m^-2 s^-1."""
    def _flux(t):
        return 1e20
    return _flux


@pytest.fixture
def zero_flux_fn():
    """Zero flux function."""
    def _flux(t):
        return 0.0
    return _flux
