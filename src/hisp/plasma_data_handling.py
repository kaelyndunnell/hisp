"""
Plasma Data Handling - imported from PFC-Tritium-Transport

This module provides a bridge to import PlasmaDataHandling from PFC-Tritium-Transport.
All data management logic should be implemented in PFC-Tritium-Transport.
"""
import os
import sys
from pathlib import Path

_pfc_tt_path = os.environ.get("PFC_TT_PATH") or os.environ.get("HISP_PFC_TT_PATH")
if _pfc_tt_path:
    sys.path.insert(0, str(Path(_pfc_tt_path).resolve()))

# Import PlasmaDataHandling from PFC-Tritium-Transport
try:
    from plasma_data_handling import PlasmaDataHandling
except ImportError as e:
    raise ImportError(
        "Could not import from PFC-Tritium-Transport. "
        "Set PFC_TT_PATH to your local clone: "
        "`conda env config vars set PFC_TT_PATH=/path/to/PFC-Tritium-Transport`"
    ) from e

# Re-export for backwards compatibility
__all__ = ['PlasmaDataHandling']
