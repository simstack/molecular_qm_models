#was moved to the QM RESULT CLASS 
# """Orbital energies from a QM (e.g. orca) calculation.
# """
# #do not use annotations if not necessary- always makes issues with the node creation table
# #from __future__ import annotations

# from typing import Optional, Dict, TYPE_CHECKING

# import logging
# import numpy as np
# import pandas as pd
# from odmantic import Model, Field

# from simstack.models import simstack_model
# from simstack.models.simple_table import SimpleTable
# from simstack.util.ui_tools import ui_hide_fields

# if TYPE_CHECKING:  # pragma: no cover - only for type checking, avoids cycles
#     from applications.electronic_structure.orca.pyorca import OrcaRun

# logger = logging.getLogger(__name__)


# @simstack_model
# class QMResult_orbital_energies(Model):
#     """Orbital energies from a QM (e.g. orca) calculation. 
#     Saves and displays both the full table and for convenience the absolute HOMO, absolute LUMO, and HOMO_LUMO_GAP
#     """

#     field_name: str = "QMResult_orbital_energies"
#     HOMO_value_eV: Optional[float] = Field(default=None)
#     LUMO_value_eV: Optional[float] = Field(default=None)
#     HOMO_LUMO_gap_eV: Optional[float] = Field(default=None)
#     HOMO_value_Hartree: Optional[float] = Field(default=None)
#     LUMO_value_Hartree: Optional[float] = Field(default=None)
#     HOMO_LUMO_gap_Hartree: Optional[float] = Field(default=None)

#     # Tabular views for GUI display (1-row SimpleTables).
#     orbital_energies_table_eV: Optional[SimpleTable] = Field(
#         default=None
#     )
#     orbital_energies_hartree: Optional[SimpleTable] = Field(
#         default=None
#     )

#     @classmethod
#     def ui_base_schema(cls):
#         """Base UI schema ordering for electronic properties.

#         We mirror the layout used in ``QMResult.ui_base_schema`` but keep it
#         focused entirely on electronic properties. Raw dict fields are hidden
#         where table-based representations exist, so GUIs can show compact
#         key/value tables by default while still having access to the
#         underlying mappings if needed.
#         """

#         base_schema = {
#             "ui:order": [
#                 "HOMO_value_eV",
#                 "LUMO_value_eV",
#                 "HOMO_LUMO_gap_eV",
#                 "HOMO_value_Hartree",
#                 "LUMO_value_Hartree",
#                 "HOMO_LUMO_gap_Hartree",
#                 "orbital_energies_table_eV",
#                 "orbital_energies_hartree",
#                 "id",
#             ],
#         }

#         hidden = [
#         ]

#         base_schema = ui_hide_fields(base_schema, hidden)
#         return base_schema

#     @classmethod
#     def from_orbital_energies_dataframe(
#         cls,
#         df: pd.DataFrame,
#     ) -> "QMResult_orbital_energies":
#         """Build a QMResult_orbital_energies instance from a parsed orbital energies DataFrame.

#         Parameters:
#             df: DataFrame with columns:
#                 - orbital_no: Orbital number
#                 - occupation: Occupation number (0.0 or 2.0)
#                 - energy_hartree: Energy in Hartree
#                 - energy_ev: Energy in eV
#                 - orbital_type: 'occupied' or 'virtual'
#                 - is_homo: bool (optional)
#                 - is_lumo: bool (optional)

#         Returns:
#             QMResult_orbital_energies: Populated instance with HOMO, LUMO, gap and tables.
#         """
#         homo_value_eV: Optional[float] = None
#         lumo_value_eV: Optional[float] = None
#         homo_lumo_gap_eV: Optional[float] = None

#         #hartree
#         homo_value_hartree: Optional[float] = None
#         lumo_value_hartree: Optional[float] = None
#         homo_lumo_gap_hartree: Optional[float] = None

#         # Local helper to build a SimpleTable from a list of dicts
#         def _build_table(df: pd.DataFrame, name: str) -> Optional[SimpleTable]:
#             if df is None or df.empty:
#                 return None
#             try:
#                 table = SimpleTable(name=name)
#                 table.add_column("orbital_no", "int")
#                 table.add_column("occupation", "float")
#                 table.add_column("energy", "float")
#                 table.add_column("orbital_type", "string")
#                 for _, row in df.iterrows():
#                     table.add_row(
#                         {
#                             "orbital_no": int(row["orbital_no"]),
#                             "occupation": float(row["occupation"]),
#                             "energy": float(row["energy"]),
#                             "orbital_type": str(row.get("orbital_type", "unknown")),
#                         }
#                     )
#                 return table
#             except Exception as e_tbl:
#                 logger.warning("Failed to build orbital energies table '%s': %s", name, e_tbl)
#                 return None

#         # Find HOMO and LUMO from the DataFrame (both eV and Hartree)
#         if df is not None and not df.empty:
#             if "is_homo" in df.columns:
#                 homo_rows = df[df["is_homo"] == True]
#                 if not homo_rows.empty:
#                     homo_value_eV = float(homo_rows.iloc[0]["energy_ev"])
#                     homo_value_hartree = float(homo_rows.iloc[0]["energy_hartree"])

#             if "is_lumo" in df.columns:
#                 lumo_rows = df[df["is_lumo"] == True]
#                 if not lumo_rows.empty:
#                     lumo_value_eV = float(lumo_rows.iloc[0]["energy_ev"])
#                     lumo_value_hartree = float(lumo_rows.iloc[0]["energy_hartree"])

#             if homo_value_eV is not None and lumo_value_eV is not None:
#                 homo_lumo_gap_eV = lumo_value_eV - homo_value_eV
#                 homo_lumo_gap_hartree = lumo_value_hartree - homo_value_hartree

#         # Build eV table (without the is_homo/is_lumo columns)
#         ev_df = df[["orbital_no", "occupation", "energy_ev", "orbital_type"]].copy()
#         ev_df = ev_df.rename(columns={"energy_ev": "energy"})
#         orbital_energies_table_eV = _build_table(ev_df, "Orbital energies (eV)")

#         # Build Hartree table
#         har_df = df[["orbital_no", "occupation", "energy_hartree", "orbital_type"]].copy()
#         har_df = har_df.rename(columns={"energy_hartree": "energy"})
#         orbital_energies_hartree = _build_table(har_df, "Orbital energies (Hartree)")

#         return cls(
#             HOMO_value_eV=homo_value_eV,
#             LUMO_value_eV=lumo_value_eV,
#             HOMO_LUMO_gap_eV=homo_lumo_gap_eV,
#             HOMO_value_Hartree=homo_value_hartree,
#             LUMO_value_Hartree=lumo_value_hartree,
#             HOMO_LUMO_gap_Hartree=homo_lumo_gap_hartree,
#             orbital_energies_table_eV=orbital_energies_table_eV,
#             orbital_energies_hartree=orbital_energies_hartree,
#         )

#was moved to the QM RESULT CLASS
