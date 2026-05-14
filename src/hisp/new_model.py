"""
New model class for CSV-driven bin simulations using new_mb_model.

This module provides a NewModel class that:
1. Uses new_mb_model.py for dynamic FESTIM model creation
2. Manages bin simulations with CSV-based configuration
3. Handles adaptive timestepping based on scenario pulses
"""

from typing import Dict, Tuple, Callable
import numpy as np

from hisp.plasma_data_handling import PlasmaDataHandling
from hisp.scenario import Scenario
from hisp.festim_models.new_mb_model import make_model_with_scenario

import festim as F


class NewModel:
    """
    Model runner that uses new_mb_model for dynamic FESTIM model creation.
    """
    
    def __init__(
        self,
        reactor,  # Reactor type from csv_bin
        scenario: Scenario,
        plasma_data_handling: PlasmaDataHandling,
        coolant_temp: float = 343.0,
        bins_meshes: Dict = None,
        temperature_model_overrides: Dict[str, Callable] | None = None,
    ):
        """
        Initialize the model runner.
        
        Args:
            reactor: Reactor object containing all bins
            scenario: Scenario with pulse sequence
            plasma_data_handling: Plasma data handler for flux/heat
            coolant_temp: Coolant temperature (K)
            bins_meshes: Dictionary of MeshBin objects keyed by bin_id (optional, defaults to BINS_MESHES)
            temperature_model_overrides: Optional mapping of material name to custom
                temperature callable; only provided materials are overridden
        """
        self.reactor = reactor
        self.scenario = scenario
        self.plasma_data_handling = plasma_data_handling
        self.coolant_temp = coolant_temp
        self.bins_meshes = bins_meshes if bins_meshes is not None else {}
        self.temperature_model_overrides = temperature_model_overrides if temperature_model_overrides is not None else {}
        
    def run_bin(self, bin, exports: bool = False, folder: str = None) -> Tuple[F.HydrogenTransportProblem, Dict]:
        """
        Run a FESTIM simulation for a single bin.
        
        Args:
            bin: Bin object to simulate
            exports: Whether to export XDMF files
            folder: Output folder for VTX checkpoint files. If None, defaults to
                    results_bin_{bin.bin_number} in the current working directory.
            
        Returns:
            Tuple of (festim_model, quantities_dict)
        """
        print(f"\n{'='*60}")
        print(f"Running bin flux_id={bin.flux_id} (sim_id={bin.sim_id})")
        print(f"  Material: {bin.material.name}")
        print(f"  Mode: {bin.mode}")
        print(f"  Location: {bin.location}")
        print(f"  Thickness: {bin.thickness*1e3:.2f} mm")
        print(f"  Surface area: {bin.surface_area:.4f} m²")
        print(f"{'='*60}")
        
        # Get mesh for this bin if available
        mesh = None
        if bin.sim_id in self.bins_meshes:
            mesh = self.bins_meshes[bin.sim_id].mesh
        
        # Set up milestones for adaptive timestepping (before model creation for profile exports)
        bin_config = bin.bin_configuration
        initial_stepsize = 1e-3 if bin.material.name == "B" else 1e-2
        milestones = self._make_milestones(initial_stepsize)
        milestones.append(self.scenario.get_maximum_time())  # Include final time for both milestones and profile export
        
        # Create FESTIM model using new_mb_model
        try:
            my_model, quantities = make_model_with_scenario(
                bin=bin,
                scenario=self.scenario,
                plasma_data_handling=self.plasma_data_handling,
                coolant_temp=self.coolant_temp,
                mesh=mesh,
                exports=exports,
                profile_export=True,
                milestones=milestones,
                folder=folder,
                temperature_model_overrides=self.temperature_model_overrides,
            )
        except Exception as e:
            print(f"ERROR: Failed to create model for bin {bin.bin_number}: {e}")
            raise
        
        # Add derived quantities to model (skip if already in exports)
        my_model.exports = my_model.exports if hasattr(my_model, 'exports') and my_model.exports else []
        for qty_name, qty in quantities.items():
            if qty not in my_model.exports:
                my_model.exports.append(qty)
        
        # Apply milestones to model (already includes final_time)
        my_model.settings.stepsize.milestones = milestones
        
        # Adaptivity settings
        my_model.settings.stepsize.growth_factor = 1.1
        my_model.settings.stepsize.cutback_factor = 0.3
        my_model.settings.stepsize.target_nb_iterations = 4
        
        # Apply max_stepsize from CSV bin configuration
        # - During FP active phase (ramp+steady+ramp): fp_max_stepsize
        # - During FP waiting / BAKE / other: max_stepsize_no_fp
        fp_max_dt = bin_config.fp_max_stepsize
        no_fp_max_dt = bin_config.max_stepsize_no_fp
        scenario_ref = self.scenario
        
        def max_stepsize_function(t):
            pulse = scenario_ref.get_pulse(t)
            if pulse.pulse_type in ("FP", "FP_D"):
                t_rel = t - scenario_ref.get_time_start_current_pulse(t)
                relative_time = t_rel % pulse.total_duration
                if relative_time < pulse.duration_no_waiting:
                    return fp_max_dt
            return no_fp_max_dt
        
        my_model.settings.stepsize.max_stepsize = max_stepsize_function
        print(f"[model] Max stepsize: FP={fp_max_dt:.1f} s, no-FP/BAKE={no_fp_max_dt:.1f} s")
        
        # Initialize and run
        print(f"Initializing FESTIM model...")
        my_model.initialise()
        
        print(f"Running simulation (final_time={self.scenario.get_maximum_time():.0f} s)...")
        my_model.run()
        
        print(f"✓ Simulation complete for bin {bin.bin_number}")
        
        return my_model, quantities
    
    def _make_milestones(self, initial_stepsize_value: float):
        """
        Build stepsize/adaptivity milestones from scenario pulses.
        (Same logic as original file, preserved for stability of runs.)
        """
        milestones = []
        current_time = 0.0

        for pulse in self.scenario.pulses:
            start_of_pulse = self.scenario.get_time_start_current_pulse(current_time)
            for i in range(pulse.nb_pulses):
                # small milestone right after each sub-pulse start
                milestones.append(start_of_pulse + pulse.total_duration * i + initial_stepsize_value)

                # ramp-up / ramp-down edges
                if i == 0:
                    milestones.append(start_of_pulse + pulse.ramp_up)
                    milestones.append(start_of_pulse + pulse.ramp_up + pulse.steady_state)
                else:
                    milestones.append(start_of_pulse + pulse.total_duration * (i - 1) + pulse.ramp_up)
                    milestones.append(start_of_pulse + pulse.total_duration * (i - 1) + pulse.ramp_up + pulse.steady_state)

                # start of next sub-pulse
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1))

                # before the end of waiting period
                assert pulse.total_duration - pulse.duration_no_waiting >= 10
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1) - 10)
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1) - 2)

                # start of waiting for this sub-pulse
                milestones.append(start_of_pulse + pulse.total_duration * i + pulse.duration_no_waiting)

                # RISP special anchor
                if getattr(pulse, "pulse_type", None) == "RISP":
                    t_begin_real_pulse = start_of_pulse + 95
                    milestones.append(t_begin_real_pulse + pulse.total_duration * i)
                    milestones.append(t_begin_real_pulse + pulse.total_duration * i + 0.001)

                # Bake+GDC: add milestones at GDC sub-pulse boundaries
                if getattr(pulse, "pulse_type", None) == "Bake+GDC":
                    gdc_ru = getattr(pulse, "gdc_ramp_up", None)
                    gdc_ss = getattr(pulse, "gdc_steady_state", None)
                    gdc_rd = getattr(pulse, "gdc_ramp_down", None)
                    if gdc_ru is not None and gdc_ss is not None and gdc_rd is not None:
                        sub_start = start_of_pulse + pulse.total_duration * i
                        milestones.append(sub_start + gdc_ru)  # end of GDC ramp-up
                        milestones.append(sub_start + gdc_ru + gdc_ss)  # end of GDC steady
                        milestones.append(sub_start + gdc_ru + gdc_ss + gdc_rd)  # end of GDC ramp-down
                        milestones.append(sub_start + gdc_ru + gdc_ss + gdc_rd + initial_stepsize_value)  # restart after GDC off

            current_time = start_of_pulse + pulse.total_duration * pulse.nb_pulses

        return sorted(np.unique(milestones).tolist())
