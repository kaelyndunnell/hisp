# HISP

Hydrogen Inventory Simulations for PFCs (HISP) is a series of code that uses FESTIM to simulate deuterium and tritium inventories in a fusion tokamak first wall and divertor PFCs.

**This fork** reworks HISP to serve as the FESTIM orchestrator for [PFC-Tritium-Transport](https://github.com/iterorganization/PFC-Tritium-Transport) (PFC-TT): it receives bin objects, material properties, scenarios, and plasma data from PFC-TT and builds the corresponding FESTIM simulations. PFC-TT owns all physics definitions, inputs, and user control.

That said, once the core PFC-TT classes are importable (by having a local clone and its conda environment), HISP can also be used independently of PFC-TT's workflow. The [`example/`](example/) folder demonstrates this: bins and scenarios are defined directly in Python and passed to HISP to build and solve a simulation.

For more advanced use cases — including flexible input handling, detailed physics control, or large parameter studies — users are encouraged to work directly with https://github.com/iterorganization/PFC-Tritium-Transport, which provides a more complete and streamlined way to set up and run simulations.

## What HISP Does

For each bin it constructs a FESTIM simulation: it translates the bin geometry (thickness, optional Cu layer, surface area) into the model domain, assigns material parameters and trap definitions, builds time-dependent boundary conditions and implantation source expressions, and selects appropriate boundary-condition types (Robin/Neumann) before assembling and solving the transport equations with adaptive time-stepping. The per-bin outputs (retained inventory, surface fluxes, concentration profiles, and time traces) are exported to JSON for post-processing or aggregation across bins.

## Dependencies

HISP depends on:
- [FESTIM](https://github.com/festim-dev/festim) — finite element solver for hydrogen transport
- [PFC-Tritium-Transport](https://github.com/iterorganization/PFC-Tritium-Transport) — provides bin definitions, material classes, scenario handling, and plasma data

Both are installed as part of the PFC-TT conda environment (see below).

## How to Install

Follow the full installation instructions in the [PFC-Tritium-Transport README](https://github.com/iterorganization/PFC-Tritium-Transport). In summary:

### 1. Clone PFC-Tritium-Transport
```bash
git clone --branch main https://github.com/iterorganization/PFC-Tritium-Transport.git
```

### 2. Create the conda environment
This step installs all core simulation dependencies including **FESTIM** (required by both PFC-Tritium-Transport and HISP), FEniCS-DOLFINx, PETSc, and all other required packages:
```bash
conda config --set channel_priority flexible
conda env create -f PFC-TT.yml
conda activate PFC-TT
```

### 3. Register the PFC-TT path
HISP needs to know where PFC-Tritium-Transport is located on your system at runtime. Register the path once in your conda environment (replace with your actual clone location):
```bash
conda env config vars set PFC_TT_PATH="/path/to/your/PFC-Tritium-Transport"
conda deactivate && conda activate PFC-TT
```

You can verify it was set correctly with:
```bash
conda env config vars list
```

### 4. Install HISP
HISP is installed without dependencies since FESTIM and all other requirements are already provided by the PFC-TT conda environment:
```bash
pip install --no-deps git+https://github.com/AdriaLlealS/hisp.git@main
pip install h_transport_materials
```

## Examples

The `example/` folder contains a working example based on the HISP paper ([Dunnell et al., 2026](https://arxiv.org/abs/2604.04751)), demonstrating a full simulation of deuterium and tritium retention in iter's first wall and divertor plasma-facing components.

This example includes the results for a single bin under a specific scenario, along with a simple plotting script to visualise the outputs and the corresponding generated plot.

See [`example/README.md`](example/README.md) for a full description of the example and instructions on how to run it.

## Running Tests

With the conda environment active and `PFC_TT_PATH` set:
```bash
cd /path/to/hisp
python -m pytest tests/ -v
```

## Development History

The first prototype (2024) was a single framework that combined simulation setup and solver execution, using MHIMS as the hydrogen isotope transport solver.

In early 2025, MHIMS was replaced by FESTIM for improved computational efficiency, and the codebase was split into two repositories — PFC-TT and HISP. During this stage, HISP still retained a significant role in handling physics definitions and simulation inputs alongside solver-related functionalities.

A subsequent major refactor further clarified the separation of responsibilities: all physics definitions and input handling were moved into PFC-TT, where they were also extended and further developed. At the same time, HISP evolved towards a role focused on solver execution and orchestration, with improved flexibility and integration with FESTIM. These developments led to the present versions of both codes.