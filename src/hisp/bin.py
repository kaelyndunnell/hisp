"""
CSV-driven bin classes - imported from PFC-Tritium-Transport
This module intentionally avoids defining any legacy HISP bin classes.
"""
# =============================================================================
# CSV-driven bin classes - imported from PFC-Tritium-Transport
# =============================================================================
import os
import sys
from pathlib import Path

_pfc_tt_path = os.environ.get("PFC_TT_PATH") or os.environ.get("HISP_PFC_TT_PATH")
if _pfc_tt_path:
    sys.path.insert(0, str(Path(_pfc_tt_path).resolve()))

# Import CSV bin classes from PFC-Tritium-Transport
# This avoids duplication — all changes should be made in PFC-Tritium-Transport/csv_bin.py
try:
    from bins_from_csv.csv_bin import BinConfiguration, Bin, BinCollection, Reactor
except ImportError as e:
    raise ImportError(
        "Could not import from PFC-Tritium-Transport. "
        "Set PFC_TT_PATH to your local clone: "
        "`conda env config vars set PFC_TT_PATH=/path/to/PFC-Tritium-Transport`"
    ) from e

# Also import the Material class from the PFC-Tritium-Transport package so
# HISP code can reference materials via `hisp.bin.Material` in the same way
# it references the CSV-driven Bin classes above.
try:
    from materials.materials import Material
except ImportError as e:
    raise ImportError(
        "Could not import from PFC-Tritium-Transport. "
        "Set PFC_TT_PATH to your local clone: "
        "`conda env config vars set PFC_TT_PATH=/path/to/PFC-Tritium-Transport`"
    ) from e

# For backwards compatibility, re-export the imported classes
# Re-export the new names
__all__ = ['BinConfiguration', 'Bin', 'BinCollection', 'Reactor', 'Material']

# These classes are imported from PFC-Tritium-Transport/csv_bin.py
# =============================================================================

# NOTE: Monkeypatch removed - new code expects bin.material to be a Material object
# Legacy code that expects bin.material to be a string should use bin.material_name instead
