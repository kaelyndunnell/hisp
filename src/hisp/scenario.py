"""
Scenario and Pulse are defined in PFC-Tritium-Transport to keep CSV-driven
scenario definitions in the PFC repository. This module acts as a shim and
re-exports `Scenario` and `Pulse` from PFC-Tritium-Transport/scenario.py.

The import strategy mirrors `hisp.bin`'s approach: try an env var `PFC_TT_PATH`,
then insert the repo root into `sys.path` so we can
`from scenario import Scenario, Pulse`.
"""
import os
import sys
from pathlib import Path

_pfc_tt_path = os.environ.get("PFC_TT_PATH") or os.environ.get("HISP_PFC_TT_PATH")
if _pfc_tt_path:
    sys.path.insert(0, str(Path(_pfc_tt_path).resolve()))

try:
    from scenario import Scenario, Pulse
except ImportError as e:
    raise ImportError(
        "Could not import from PFC-Tritium-Transport. "
        "Set PFC_TT_PATH to your local clone: "
        "`conda env config vars set PFC_TT_PATH=/path/to/PFC-Tritium-Transport`"
    ) from e

__all__ = ["Scenario", "Pulse"]
