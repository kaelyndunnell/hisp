from hisp.plamsa_data_handling import PlasmaDataHandling
from hisp.scenario import Scenario
from hisp.bin import Reactor, SubBin, DivBin
from hisp.helpers import periodic_step_function
from hisp.festim_models import (
    make_W_mb_model,
    make_B_mb_model,
    make_DFW_mb_model,
    make_temperature_function,
    make_particle_flux_function,
)

import numpy as np
from typing import List


class Model:
    """
    The main HISP model class.
    Takes a reactor, a scenario, a plasma data handling object and runs the FESTIM model(s) for each reactor bin.
    """

    def __init__(
        self,
        reactor: Reactor,
        scenario: Scenario,
        plasma_data_handling: PlasmaDataHandling,
        coolant_temp: float = 343.0,
    ):
        """
        Args:
            reactor: The reactor to run the model for
            scenario: The scenario to run the model for
            plasma_data_handling: The plasma data handling object
            coolant_temp: The coolant temperature (K)
        """
        self.reactor = reactor
        self.scenario = scenario
        self.plasma_data_handling = plasma_data_handling
        self.coolant_temp = coolant_temp

    def run_bin(self, bin: SubBin | DivBin):
        """
        Runs the FESTIM model for the given bin.

        Args:
            bin: The bin to run the model for

        Returns:
            The model, the quantities
        """
        # create the FESTIM model
        my_model, quantities = self.which_model(bin)

        # add milestones for stepsize and adaptivity
        milestones = self.make_milestones()
        milestones.append(my_model.settings.final_time)
        my_model.settings.stepsize.milestones = milestones

        # add adatpivity settings
        my_model.settings.stepsize.growth_factor = 1.2
        my_model.settings.stepsize.cutback_factor = 0.9
        my_model.settings.stepsize.target_nb_iterations = 4

        # add the stepsize cap function
        my_model.settings.stepsize.max_stepsize = self.max_stepsize

        # run the model
        my_model.initialise()
        my_model.run()

        return my_model, quantities

    def which_model(self, bin: SubBin | DivBin):
        """Returns the correct model for the subbin.

        Args:
            bin: The bin/subbin to get the model for

        Returns:
            The model and the quantities to plot
        """
        temperature_fuction = make_temperature_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            coolant_temp=self.coolant_temp,
        )
        d_ion_incident_flux = make_particle_flux_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            ion=True,
            tritium=False,
        )
        tritium_ion_flux = make_particle_flux_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            ion=True,
            tritium=True,
        )
        deuterium_atom_flux = make_particle_flux_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            ion=False,
            tritium=False,
        )
        tritium_atom_flux = make_particle_flux_function(
            scenario=self.scenario,
            plasma_data_handling=self.plasma_data_handling,
            bin=bin,
            ion=False,
            tritium=True,
        )
        common_args = {
            "deuterium_ion_flux": d_ion_incident_flux,
            "tritium_ion_flux": tritium_ion_flux,
            "deuterium_atom_flux": deuterium_atom_flux,
            "tritium_atom_flux": tritium_atom_flux,
            "final_time": self.scenario.get_maximum_time() - 1,
            "temperature": temperature_fuction,
            "L": bin.thickness,
        }

        if isinstance(bin, DivBin):
            parent_bin_index = bin.index
        elif isinstance(bin, SubBin):
            parent_bin_index = bin.parent_bin_index

        if bin.material == "W":
            return make_W_mb_model(
                **common_args,
                folder=f"mb{parent_bin_index+1}_{bin.mode}_results",
            )
        elif bin.material == "B":
            return make_B_mb_model(
                **common_args,
                folder=f"mb{parent_bin_index+1}_{bin.mode}_results",
            )
        elif bin.material == "SS":
            return make_DFW_mb_model(
                **common_args,
                folder=f"mb{parent_bin_index+1}_dfw_results",
            )
        else:
            raise ValueError(f"Unknown material: {bin.material} for bin {bin.index}")

    def max_stepsize(self, t: float) -> float:
        pulse = self.scenario.get_pulse(t)
        relative_time = t - self.scenario.get_time_start_current_pulse(t)
        return periodic_step_function(
            relative_time,
            period_on=pulse.duration_no_waiting,
            period_total=pulse.total_duration,
            value=pulse.duration_no_waiting / 10,
            value_off=None,
        )

    def make_milestones(self) -> List[float]:
        """
        Returns the milestones for the stepsize.
        For each pulse, the milestones are the start
        of the pulse, the start of the waiting period,
        and the end of the pulse.

        Returns:
            The milestones in seconds
        """
        milestones = []
        current_time = 0
        for pulse in self.scenario.pulses:
            start_of_pulse = self.scenario.get_time_start_current_pulse(current_time)
            for i in range(pulse.nb_pulses):
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1))
                milestones.append(
                    start_of_pulse
                    + pulse.total_duration * i
                    + pulse.duration_no_waiting
                )

            current_time = start_of_pulse + pulse.total_duration * pulse.nb_pulses

        return sorted(np.unique(milestones))
