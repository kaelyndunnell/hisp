"""Tests for temperature calculation functions.

Verifies the 1D steady-state temperature profile calculations for:
- Tungsten (W) plasma-facing components with/without Cu interlayer
- Stainless steel (SS) components
- Boron (B) surface temperature correlation
- Underlying tungsten_slab_temperature analytical solution
"""
import pytest
import numpy as np
from hisp.festim_models.new_mb_model import (
    calculate_temperature_W,
    calculate_temperature_SS,
    calculate_temperature_B,
    tungsten_slab_temperature,
)


class TestCalculateTemperatureW:
    def test_linear_in_x(self):
        """W temperature varies linearly through slab thickness (constant k)."""
        x = np.linspace(0, 6e-3, 50)
        T = calculate_temperature_W(x, heat_flux=5e6, coolant_temp=343.0,
                                    thickness=6e-3, copper_thickness=1e-3)
        # Linear fit: residuals should be negligible
        coeffs = np.polyfit(x, T, 1)
        T_fit = np.polyval(coeffs, x)
        assert np.allclose(T, T_fit, atol=1e-6)

    def test_increases_with_heat_flux(self):
        """Higher heat flux produces higher surface temperature."""
        x = np.array([0.0])
        T_low = calculate_temperature_W(x, heat_flux=1e6, coolant_temp=343.0,
                                        thickness=6e-3, copper_thickness=1e-3)
        T_high = calculate_temperature_W(x, heat_flux=10e6, coolant_temp=343.0,
                                         thickness=6e-3, copper_thickness=1e-3)
        assert T_high > T_low

    def test_with_copper_thickness(self):
        """With Cu interlayer, uses 2-layer analytical slab model."""
        x = np.array([0.0])
        T = calculate_temperature_W(x, heat_flux=5e6, coolant_temp=343.0,
                                    thickness=6e-3, copper_thickness=1e-3)
        assert T > 343.0

    def test_without_copper_thickness(self):
        """Without Cu, uses Remi's linear correlation T = a*q + T_cool."""
        x = np.array([0.0])
        T = calculate_temperature_W(x, heat_flux=5e6, coolant_temp=343.0,
                                    thickness=6e-3, copper_thickness=None)
        # Surface temp: 1.1e-4 * 5e6 + 343 = 893 K
        assert T == pytest.approx(1.1e-4 * 5e6 + 343.0)

    def test_zero_flux_returns_coolant_temp(self):
        """Zero heat flux: surface temperature equals coolant temperature."""
        x = np.array([0.0])
        T = calculate_temperature_W(x, heat_flux=0.0, coolant_temp=400.0,
                                    thickness=6e-3, copper_thickness=None)
        assert T == pytest.approx(400.0)


class TestCalculateTemperatureSS:
    def test_linear_in_x(self):
        """SS temperature varies linearly through slab (constant k)."""
        x = np.linspace(0, 3e-3, 50)
        T = calculate_temperature_SS(x, heat_flux=3.5e5, coolant_temp=343.0,
                                     thickness=3e-3)
        coeffs = np.polyfit(x, T, 1)
        T_fit = np.polyval(coeffs, x)
        assert np.allclose(T, T_fit, atol=1e-6)

    def test_zero_flux_gives_coolant_temp(self):
        """Zero heat flux: entire slab at coolant temperature."""
        x = np.linspace(0, 3e-3, 10)
        T = calculate_temperature_SS(x, heat_flux=0.0, coolant_temp=343.0,
                                     thickness=3e-3)
        assert np.allclose(T, 343.0)

    def test_surface_hotter_than_rear(self):
        """Plasma-facing surface (x=0) is hotter than coolant side."""
        thickness = 3e-3
        T_front = calculate_temperature_SS(0.0, heat_flux=3.5e5, coolant_temp=343.0,
                                           thickness=thickness)
        T_rear = calculate_temperature_SS(thickness, heat_flux=3.5e5, coolant_temp=343.0,
                                          thickness=thickness)
        assert T_front > T_rear


class TestCalculateTemperatureB:
    def test_increases_with_flux(self):
        """Boron surface temperature grows with applied heat flux."""
        T_low = calculate_temperature_B(heat_flux=1e5, coolant_temp=343.0)
        T_high = calculate_temperature_B(heat_flux=5e6, coolant_temp=343.0)
        assert T_high > T_low

    def test_zero_flux(self):
        """Zero flux: boron temperature equals coolant temperature."""
        T = calculate_temperature_B(heat_flux=0.0, coolant_temp=343.0)
        assert T == pytest.approx(343.0)


class TestTungstenSlabTemperature:
    def test_surface_greater_than_interface(self):
        """W surface is hotter than W/Cu interface."""
        T_surf, T_interface = tungsten_slab_temperature(
            q_front=5e6, D_W=6e-3, D_Cu=1e-3, T_cool=343.0
        )
        assert T_surf > T_interface

    def test_zero_flux(self):
        """Zero flux: both surface and interface equal coolant temperature."""
        T_surf, T_interface = tungsten_slab_temperature(
            q_front=0.0, D_W=6e-3, D_Cu=1e-3, T_cool=343.0
        )
        assert T_surf == pytest.approx(343.0)
        assert T_interface == pytest.approx(343.0)
