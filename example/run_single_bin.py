"""Run a single FESTIM simulation for one ITER bin.

Usage:
    python run_single_bin.py [--bin-index N] [--scenario SCENARIO]

Arguments:
    --bin-index N       Index into the bins list (default: 0, first bin)
    --scenario SCENARIO Which scenario to use:
                        "do_nothing"       - 9 FP + 1 FP (long wait) + bake (default)
                        "capability_test"  - 5 FP + FP_D + 5 FP + FP_D + GDC + bake
                        "just_glow"        - 10 FP + GDC + bake

Example:
    python run_single_bin.py --bin-index 3 --scenario just_glow
"""
import argparse
import json
import os
from pathlib import Path
import importlib

# hisp shims handle PFC-TT path resolution
from hisp.new_model import NewModel
from make_iter_bins import make_iter_bins


SCENARIOS = ["do_nothing", "capability_test", "just_glow"]


def load_scenario(name):
    """Import scenario and plasma_data_handling from the scenario module."""
    module = importlib.import_module(f"scenario_{name}")
    return module.scenario, module.plasma_data_handling


def main():
    parser = argparse.ArgumentParser(description="Run a FESTIM simulation for one ITER bin.")
    parser.add_argument("--bin-index", type=int, default=0,
                        help="Index into the bins list (default: 0)")
    parser.add_argument("--scenario", type=str, default="do_nothing",
                        choices=SCENARIOS,
                        help="Scenario to use (default: do_nothing)")
    args = parser.parse_args()

    # Change to script directory so relative data paths work
    os.chdir(Path(__file__).parent)

    # Create bins
    bins = make_iter_bins()
    print(f"Created {len(bins)} bins")

    if args.bin_index < 0 or args.bin_index >= len(bins):
        raise IndexError(f"--bin-index {args.bin_index} out of range [0, {len(bins)-1}]")

    selected_bin = bins[args.bin_index]
    print(f"\nSelected bin index {args.bin_index}:")
    print(f"  flux_id={selected_bin.flux_id}, material={selected_bin.material.name}")
    print(f"  mode={selected_bin.mode}, location={selected_bin.location}")
    print(f"  thickness={selected_bin.thickness*1e3:.2f} mm")
    print(f"  BC front: {selected_bin.bin_configuration.bc_plasma_facing_surface}")
    print(f"  BC rear:  {selected_bin.bin_configuration.bc_rear_surface}")

    # Load scenario and plasma data handling
    scenario, pdh = load_scenario(args.scenario)
    print(f"\nScenario: {args.scenario}")
    print(f"  Total time: {scenario.get_maximum_time():.0f} s")
    print(f"  Pulses: {len(scenario.pulses)} types")

    # Create output folder
    output_folder = f"results_bin{args.bin_index}_{selected_bin.material.name}_{selected_bin.mode}"
    os.makedirs(output_folder, exist_ok=True)

    # Build and run
    model_runner = NewModel(
        reactor=None,
        scenario=scenario,
        plasma_data_handling=pdh,
        coolant_temp=343.0,
    )

    model, quantities = model_runner.run_bin(
        bin=selected_bin,
        exports=False,
        folder=output_folder,
    )

    # Print final inventory
    print(f"\n{'='*60}")
    print(f"Final inventory for bin {args.bin_index}:")
    for name, qty in quantities.items():
        if hasattr(qty, "data") and len(qty.data) > 0:
            val = qty.data[-1]
            if hasattr(val, '__len__'):
                continue  # skip profile arrays
            print(f"  {name}: {val:.4e}")
    print(f"{'='*60}")

    # Save results to JSON
    scalar_data = {}
    profile_data = {}
    for name, qty in quantities.items():
        if not hasattr(qty, "data") or len(qty.data) == 0:
            continue
        if name.endswith("_profile"):
            profile_data[name] = {
                "x": qty.x.tolist(),
                "t": qty.t if isinstance(qty.t, list) else list(qty.t),
                "data": [arr.tolist() for arr in qty.data],
            }
        else:
            scalar_data[name] = {"data": [float(v) for v in qty.data]}

    # Add time array and metadata
    first_scalar = next((q for n, q in quantities.items() if not n.endswith("_profile")), None)
    if first_scalar is not None and hasattr(first_scalar, "t"):
        scalar_data["t"] = [float(v) for v in first_scalar.t]
    scalar_data["metadata"] = {
        "bin_index": args.bin_index,
        "flux_id": selected_bin.flux_id,
        "material": selected_bin.material.name,
        "mode": selected_bin.mode,
        "location": selected_bin.location,
        "scenario": args.scenario,
    }

    results_file = os.path.join(output_folder, "results.json")
    with open(results_file, "w") as f:
        json.dump(scalar_data, f, indent=4)
    print(f"\nScalar results saved to {results_file}")

    if profile_data:
        profiles_file = os.path.join(output_folder, "profiles.json")
        with open(profiles_file, "w") as f:
            json.dump(profile_data, f, indent=4)
        print(f"Profile results saved to {profiles_file}")


if __name__ == "__main__":
    main()
