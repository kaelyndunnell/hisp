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
        milestones = self.make_milestones(
            initial_stepsize_value=my_model.settings.stepsize.initial_value
        )
        milestones.append(my_model.settings.final_time)
        my_model.settings.stepsize.milestones = milestones

        # add adaptivity settings
        my_model.settings.stepsize.growth_factor = 1.2
        my_model.settings.stepsize.cutback_factor = 0.9
        my_model.settings.stepsize.target_nb_iterations = 4

        # add the stepsize cap function
        if bin.material == "B":
            my_model.settings.stepsize.max_stepsize = self.B_stepsize
        else:
            my_model.settings.stepsize.max_stepsize = self.max_stepsize

        # run the model
        my_model.initialise()
        my_model.run()

        return my_model, quantities

    # TODO implement this method
    def run_all_bins(self):
        pass

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
                custom_rtol=self.bake_rtol,
                folder=f"mb{parent_bin_index}_{bin.mode}_results",
            )
        elif bin.material == "B":
            return make_B_mb_model(
                **common_args,
                custom_rtol=self.make_custom_rtol,
                folder=f"mb{parent_bin_index}_{bin.mode}_results",
            )
        elif bin.material == "SS":
            return make_DFW_mb_model(
                **common_args,
                folder=f"mb{parent_bin_index}_dfw_results",
            )
        else:
            raise ValueError(f"Unknown material: {bin.material} for bin {bin.index}")

    def max_stepsize(self, t: float) -> float:
        pulse = self.scenario.get_pulse(t)
        relative_time = t - self.scenario.get_time_start_current_pulse(t)  # Pulse()
        if pulse.pulse_type == "RISP":
            relative_time_within_sub_pulse = relative_time % pulse.total_duration
            # RISP has a special treatment
            time_real_risp_starts = (
                100  # (s) relative time at which the real RISP starts
            )
            if relative_time_within_sub_pulse < time_real_risp_starts - 11:
                value = None  # s
            elif relative_time_within_sub_pulse < time_real_risp_starts + 160:
                value = 1e-3  # s
            # elif relative_time_within_sub_pulse  < time_real_risp_starts + 1:
            #     value = 0.01  # s
            # elif relative_time_within_sub_pulse  < time_real_risp_starts + 50:
            #     value = 0.1  # s
            else:
                # NOTE this seems to have an influence on the accuracy of the calculation
                value = 1  # s
        else:
            relative_time_within_sub_pulse = relative_time % pulse.total_duration
            # the stepsize is 1/10 of the duration of the pulse
            if pulse.pulse_type == "FP":
                if relative_time_within_sub_pulse < pulse.duration_no_waiting:
                    value = 0.1  # s
                else:
                    value = pulse.duration_no_waiting / 10
            else:
                value = pulse.duration_no_waiting / 10
        return periodic_step_function(
            relative_time,
            period_on=pulse.duration_no_waiting,
            period_total=pulse.total_duration,
            value=value,
            value_off=None,
        )

    def B_stepsize(self, t: float) -> float:
        pulse = self.scenario.get_pulse(t)
        relative_time = t - self.scenario.get_time_start_current_pulse(t)  # Pulse()
        if pulse.pulse_type == "RISP":
            relative_time_within_sub_pulse = relative_time % pulse.total_duration
            # RISP has a special treatment
            time_real_risp_starts = (
                100  # (s) relative time at which the real RISP starts
            )
            if relative_time_within_sub_pulse < time_real_risp_starts - 5:
                value = None  # s
            elif relative_time_within_sub_pulse < time_real_risp_starts + 160:
                value = 1e-4  # s
            else:
                # NOTE this seems to have an influence on the accuracy of the calculation
                value = 1  # s
        else:
            relative_time_within_sub_pulse = relative_time % pulse.total_duration
            # the stepsize is 1/10 of the duration of the pulse
            if pulse.pulse_type == "FP":
                if relative_time_within_sub_pulse < pulse.duration_no_waiting:
                    value = 0.01  # s
                else:
                    value = pulse.duration_no_waiting / 10
            elif pulse.pulse_type == "BAKE":
                value = pulse.duration_no_waiting / 10
            else:
                value = pulse.duration_no_waiting / 100
        return periodic_step_function(
            relative_time,
            period_on=pulse.duration_no_waiting,
            period_total=pulse.total_duration,
            value=value,
            value_off=None,
        )

    def make_milestones(self, initial_stepsize_value: float) -> List[float]:
        """
        Returns the milestones for the stepsize.
        For each pulse, the milestones are the start
        of the pulse, the start of the waiting period,
        and the end of the pulse.

        Args:
            initial_stepsize_value: the value of the stepsize at
                the beginning of each pulse (s)

        Returns:
            The milestones in seconds
        """
        milestones = []
        current_time = 0  # initialise the current time (s)

        # loop over all pulses (or rather sequences of pulses)
        for pulse in self.scenario.pulses:

            start_of_pulse = self.scenario.get_time_start_current_pulse(current_time)

            # loop over all "pulses" whithin the current pulse
            for i in range(pulse.nb_pulses):

                # hack: a milestone right after to ensure the stepsize is small enough
                milestones.append(
                    start_of_pulse + pulse.total_duration * i + initial_stepsize_value
                )

                if i == 0:
                    # end of ramp up
                    milestones.append(start_of_pulse + pulse.ramp_up)

                    # start of ramp down
                    milestones.append(
                        start_of_pulse + pulse.ramp_up + pulse.steady_state
                    )

                else:
                    # end of ramp up
                    milestones.append(
                        start_of_pulse + pulse.total_duration * (i - 1) + pulse.ramp_up
                    )

                    # start of ramp down
                    milestones.append(
                        start_of_pulse
                        + pulse.total_duration * (i - 1)
                        + pulse.ramp_up
                        + pulse.steady_state
                    )

                # start of the next pulse
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1))

                # add milestones 10 s before the end of the waiting period
                assert pulse.total_duration - pulse.duration_no_waiting >= 10
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1) - 10)

                # add milestones 2 s before the end of the waiting period
                milestones.append(start_of_pulse + pulse.total_duration * (i + 1) - 2)
                # start of the waiting period of this pulse
                milestones.append(
                    start_of_pulse
                    + pulse.total_duration * i
                    + pulse.duration_no_waiting
                )

                # RISP pulses need additional milestones
                if pulse.pulse_type == "RISP":

                    t_begin_real_pulse = (
                        start_of_pulse + 95
                    )  # time at which the real RISP starts

                    # a milestone for when the real RISP starts
                    milestones.append(t_begin_real_pulse + pulse.total_duration * i)

                    # NOTE do we need this?
                    # hack: a milestone right after to ensure the stepsize is small enough
                    milestones.append(
                        t_begin_real_pulse + pulse.total_duration * i + 0.001
                    )

            # update the current time to the end of the current "sequence" of pulses
            current_time = start_of_pulse + pulse.total_duration * pulse.nb_pulses

        return sorted(np.unique(milestones))

    def make_custom_rtol(self, t: float) -> float:
        pulse = self.scenario.get_pulse(t)
        relative_time = t - self.scenario.get_time_start_current_pulse(t)
        if pulse.pulse_type == "GDC" or pulse.pulse_type == "ICWC":
            rtol = 1e-11
        elif pulse.pulse_type == "BAKE":
            rtol = 1e-13
        elif pulse.pulse_type == "FP":
            # rtol = 1e-10
            if relative_time % pulse.total_duration > pulse.duration_no_waiting:
                rtol = 1e-12
            else:
                rtol = 1e-6
        else:
            rtol = 1e-10
        return rtol

    def bake_rtol(self, t: float) -> float:
        pulse = self.scenario.get_pulse(t)
        if pulse.pulse_type == "BAKE":
            rtol = 1e-12
        else:
            rtol = 1e-8
        return rtol
