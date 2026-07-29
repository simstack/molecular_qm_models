import os
from pathlib import Path
from pprint import pprint
from typing import Optional, List, TypedDict, TYPE_CHECKING
import asyncio
from odmantic import Model, Field
from pydantic import model_validator
import logging
import pandas as pd

from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from molecular_qm_models.molecule import MoleculeList, Molecule, Atom
if TYPE_CHECKING:
    from applications.electronic_structure.orca.pyorca import OrcaRun
from simstack.models import simstack_model
from simstack.models.files import FileStack
from simstack.models.file_list import FileList
from simstack.models.simple_table import SimpleTable
from simstack.util.project_root_finder import find_project_root
logger = logging.getLogger(__name__)


class QMResultDict(TypedDict, total=False):
    charge: int
    dipole: Optional[float]
    final_energy: Optional[float]
    dipole_moment: Optional[List[float]]
    energies: List[float]
    scf_converge: bool  # Temporary key used during parsing
    scf_converged: bool
    normal_termination: bool
    optimization_converged: bool
    scf_energies: List[float]
    files: FileList
    molecule_list: List  # Temporary key used during parsing
    structures: MoleculeList
    final_structure: Molecule
    field_name: str


@simstack_model
class QMThermoResult(Model):
    field_name: str = "QMThermoResult"
    thermodynamics_table: Optional[SimpleTable] = None
    detailed_thermo_table: Optional[SimpleTable] = None

    zpve: Optional[float] = Field(default=None)
    thermal_energy_correction: Optional[float] = Field(default=None)
    enthalpy_correction: Optional[float] = Field(default=None)
    gibbs_free_energy_correction: Optional[float] = Field(default=None)

    # All thermo fields
    E0: Optional[float] = Field(default=None)
    B: Optional[List[float]] = Field(default=None)
    sigma: Optional[int] = Field(default=None)
    T: Optional[float] = Field(default=None)
    P: Optional[float] = Field(default=None)
    
    S_elec: Optional[float] = Field(default=None)
    S_trans: Optional[float] = Field(default=None)
    S_rot: Optional[float] = Field(default=None)
    S_vib: Optional[float] = Field(default=None)
    S_tot: Optional[float] = Field(default=None)
    
    Cv_elec: Optional[float] = Field(default=None)
    Cv_trans: Optional[float] = Field(default=None)
    Cv_rot: Optional[float] = Field(default=None)
    Cv_vib: Optional[float] = Field(default=None)
    Cv_tot: Optional[float] = Field(default=None)
    
    Cp_elec: Optional[float] = Field(default=None)
    Cp_trans: Optional[float] = Field(default=None)
    Cp_rot: Optional[float] = Field(default=None)
    Cp_vib: Optional[float] = Field(default=None)
    Cp_tot: Optional[float] = Field(default=None)
    
    E_elec: Optional[float] = Field(default=None)
    E_trans: Optional[float] = Field(default=None)
    E_rot: Optional[float] = Field(default=None)
    E_vib: Optional[float] = Field(default=None)
    E_tot: Optional[float] = Field(default=None)
    
    H_elec: Optional[float] = Field(default=None)
    H_trans: Optional[float] = Field(default=None)
    H_rot: Optional[float] = Field(default=None)
    H_vib: Optional[float] = Field(default=None)
    H_tot: Optional[float] = Field(default=None)
    
    G_elec: Optional[float] = Field(default=None)
    G_trans: Optional[float] = Field(default=None)
    G_rot: Optional[float] = Field(default=None)
    G_vib: Optional[float] = Field(default=None)
    G_tot: Optional[float] = Field(default=None)
    
    ZPE_elec: Optional[float] = Field(default=None)
    ZPE_trans: Optional[float] = Field(default=None)
    ZPE_rot: Optional[float] = Field(default=None)
    ZPE_vib: Optional[float] = Field(default=None)
    ZPE_tot: Optional[float] = Field(default=None)

    ZPE_corr: Optional[float] = Field(default=None)
    E_corr: Optional[float] = Field(default=None)
    H_corr: Optional[float] = Field(default=None)
    G_corr: Optional[float] = Field(default=None)

@simstack_model
class QMResult(Model):
    field_name: str = "QMResult"
    bond_orders_json: Optional[str] = Field(default=None)
    charge: int = 0
    dipole: Optional[float] = Field(default=None)
    final_energy: Optional[float] = Field(default=None)
    dipole_moment: Optional[List[float]] = Field(default=None)
    energies: List[float] = Field(default=[])
    status: Optional[str] = None
    error: Optional[str] = None
    task_status: Optional[TaskStatus] = None
    normal_termination: Optional[bool] = Field(default=None)
    optimization_converged: Optional[bool] = Field(default=None)

    scf_energies:  List[float] = Field(default_factory=list)
    scf_converged: Optional[bool] = Field(default=None)
    structures: Optional[MoleculeList] = None
    final_structure: Optional[Molecule] = None
    files: FileList = Field(default_factory=FileList)

    excited_states: Optional[SimpleTable] = None
    excited_state_transitions: Optional[SimpleTable] = None
    absorption_spectrum: Optional[SimpleTable] = None
    mayer_analysis: Optional[SimpleTable] = None
    mayer_bond_orders: Optional[SimpleTable] = None

    vibrational_frequencies: Optional[SimpleTable] = None
    normal_modes: Optional[SimpleTable] = None
    ir_spectrum: Optional[SimpleTable] = None

    enthalpy: Optional[float] = Field(default=None)
    gibbs_free_energy: Optional[float] = Field(default=None)
    entropy: Optional[float] = Field(default=None)
    internal_energy: Optional[float] = Field(default=None)

    dftmrci_configurations: Optional[SimpleTable] = None
    max_amplitude_not_in_ref_space: Optional[List[float]] = Field(default=None)

    # Hyperpolarizability (β) – typically a small table with βzzz for up to 3 frequency pairs - was all moved to elprop Result! and should stay there!
    #hyperpolarizability: Optional[SimpleTable] = None

    # Orbital energies - convenience fields for HOMO, LUMO, gap (both eV and Hartree)
    HOMO_value_eV: Optional[float] = Field(default=None)
    LUMO_value_eV: Optional[float] = Field(default=None)
    HOMO_LUMO_gap_eV: Optional[float] = Field(default=None)
    HOMO_value_Hartree: Optional[float] = Field(default=None)
    LUMO_value_Hartree: Optional[float] = Field(default=None)
    HOMO_LUMO_gap_Hartree: Optional[float] = Field(default=None)

    # Tabular views for GUI display (1-row SimpleTables).
    orbital_energies_table_eV: Optional[SimpleTable] = Field(default=None)
    orbital_energies_hartree: Optional[SimpleTable] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    async def from_orca_output(cls, orca_run: "OrcaRun", task_id: Optional[str] = None) -> "QMResult":
        """Creates an instance of the class from an OrcaRun object."""
        from applications.electronic_structure.orca.pyorca import OrcaRun
        
        def molecule_from_structure(structure) -> Molecule:
            molecule = Molecule()
            for site in structure.sites:
                atom = Atom.from_coords(element=site.label, coords=site.coords)
                molecule.add_atom(atom)
            return molecule


        logger.info(f"Reading results from run task_id: {task_id} V0.1")
        
        result_dict: QMResultDict = {
            "charge": 0,
            "dipole": 0,
            "final_energy": 0,
            "dipole_moment": [0, 0, 0],
            "energies": [],
            "scf_converge": False,
            "normal_termination": False,
            "optimization_converged": False,
            "scf_energies": [],
            "files": [],

        }

        try:
            # Copy simple scalar/vector quantities from the OrcaRun object.
            # All electronic-property related keys have been removed from
            # result_dict, so a simple generic copy is sufficient here.
            for key in list(result_dict.keys()):
                result_dict[key] = getattr(orca_run, key, result_dict[key])

            # Fix historical typo in the attribute name
            result_dict["scf_converged"] = result_dict["scf_converge"]
            del result_dict["scf_converge"]


        except Exception as e:
            logger.info(f"Reading results from run task_id: {task_id} V0.1 --- Values FAILED {e}")

        files_dict = {
            "engrad_file": "orca.engrad",
            "opt_file": "orca.opt",
            "property_file": "orca.property.txt",
            "xyz_file": "orca.xyz",
            "gbw_file": "orca.gbw",
            "densities": "orca.densities",
            "hessian": "orca.hess",
            "trajectory": "orca_trj.xyz",
            "dft_mrci_input": "orca.DFTMRCI.inp",
            "bkji": "orca.bkji",
        }

        try:
            final_structure = molecule_from_structure(orca_run.final_structure)
            final_structure = await context.db.save(final_structure)
            logger.info(f"Reading results from run task_id: {task_id} V0.1 --- Final structure done")
        except Exception as e:
            logger.info(f"Reading results from run task_id: {task_id} V0.1 --- Final structure FAILED")
            final_structure = Molecule()


        molecule_list = MoleculeList()
        try:
            for structure in orca_run.structures:
                molecule = molecule_from_structure(structure)
                molecule = await context.db.save(molecule)
                molecule_list.add_molecule(molecule)
            await context.db.save(molecule_list)
            logger.info(f"Reading results from run task_id: {task_id} V0.1 --- Structures done")
        except Exception as e:
            logger.info(f"Reading results from run task_id: {task_id} V0.1 --- Structures FAILED {e}")
            final_structure = Molecule()

        file_list = FileList()
        for key, file_path in files_dict.items():
            if os.path.exists(file_path):
                file_stack = FileStack.from_local_file(file_path, in_memory=True, secure_source=True)
                await context.db.save(file_stack)
                file_list.append(file_stack)

        logger.info(f"Reading results from run task_id: {task_id} V0.1 --- FileList Done")

        result_dict["structures"] = molecule_list
        result_dict["final_structure"] = final_structure

        result_dict["files"] = file_list
        #del result_dict["molecule_list"]
        result = cls(**result_dict)

        logger.info(
            f"Reading results from run task_id: {task_id} V0.1 --- Result creation done"
        )
        return result


    def set_values_from_orbital_energies_dataframe(self, df: pd.DataFrame) -> None:
        """Update orbital energy fields on this QMResult instance from a parsed DataFrame.

        This method updates (rather than replaces) the existing QMResult instance,
        only modifying the orbital energy related fields and leaving all other
        fields untouched.

        Parameters:
            df: DataFrame with columns:
                - orbital_no: Orbital number
                - occupation: Occupation number (0.0 or 2.0)
                - energy_hartree: Energy in Hartree
                - energy_ev: Energy in eV
                - orbital_type: 'occupied' or 'virtual'
                - is_homo: bool (optional)
                - is_lumo: bool (optional)
        """
        # Local helper to build a SimpleTable from a DataFrame
        def _build_table(df: pd.DataFrame, name: str) -> Optional[SimpleTable]:
            if df is None or df.empty:
                return None
            try:
                table = SimpleTable(name=name)
                table.add_column("orbital_no", "int")
                table.add_column("occupation", "float")
                table.add_column("energy", "float")
                table.add_column("orbital_type", "string")
                for _, row in df.iterrows():
                    table.add_row(
                        {
                            "orbital_no": int(row["orbital_no"]),
                            "occupation": float(row["occupation"]),
                            "energy": float(row["energy"]),
                            "orbital_type": str(row.get("orbital_type", "unknown")),
                        }
                    )
                return table
            except Exception as e_tbl:
                logger.warning("Failed to build orbital energies table '%s': %s", name, e_tbl)
                return None

        if df is None or df.empty:
            logger.warning("Empty or None DataFrame provided to set_values_from_orbital_energies_dataframe")
            return self # we should always return the instance back even if we do not update anything because there is nothing to update

        # Find HOMO and LUMO from the DataFrame (both eV and Hartree)
        # HOMO: last occupied orbital (OCC > 0)
        # LUMO: first virtual orbital (OCC == 0), which is the orbital immediately after the HOMO

        homo_value_eV: Optional[float] = None
        lumo_value_eV: Optional[float] = None
        homo_value_hartree: Optional[float] = None
        lumo_value_hartree: Optional[float] = None

        if "occupation" not in df.columns:
            logger.warning("DataFrame missing 'occupation' column, cannot determine HOMO/LUMO")
        else:
            # Find the index of the HOMO (last occupied orbital, OCC > 0)
            occupied_mask = df["occupation"] > 0
            if occupied_mask.any():
                # Get the integer position of the last occupied orbital
                occupied_indices = df.index[occupied_mask]
                homo_position = df.index.get_loc(occupied_indices[-1])
                
                # HOMO is the last occupied orbital
                homo_row = df.iloc[homo_position]
                homo_value_eV = float(homo_row["energy_ev"])
                homo_value_hartree = float(homo_row["energy_hartree"])

                # LUMO is the next orbital after HOMO (first virtual orbital)
                next_position = homo_position + 1
                if next_position < len(df):
                    lumo_row = df.iloc[next_position]
                    # Verify this is indeed a virtual orbital (OCC == 0 or OCC == 1 for alpha in unrestricted)
                    if lumo_row["occupation"] == 0 or lumo_row["occupation"] == 1:
                        lumo_value_eV = float(lumo_row["energy_ev"])
                        lumo_value_hartree = float(lumo_row["energy_hartree"])

        # Update the instance fields
        self.HOMO_value_eV = homo_value_eV
        self.LUMO_value_eV = lumo_value_eV
        if homo_value_eV is not None and lumo_value_eV is not None:
            self.HOMO_LUMO_gap_eV = lumo_value_eV - homo_value_eV

        self.HOMO_value_Hartree = homo_value_hartree
        self.LUMO_value_Hartree = lumo_value_hartree
        if homo_value_hartree is not None and lumo_value_hartree is not None:
            self.HOMO_LUMO_gap_Hartree = lumo_value_hartree - homo_value_hartree

        # Build and assign the tabular views for GUI display
        # Build eV table (without the is_homo/is_lumo columns)
        ev_df = df[["orbital_no", "occupation", "energy_ev", "orbital_type"]].copy()
        ev_df = ev_df.rename(columns={"energy_ev": "energy"})
        self.orbital_energies_table_eV = _build_table(ev_df, "Orbital energies (eV)")

        # Build Hartree table
        har_df = df[["orbital_no", "occupation", "energy_hartree", "orbital_type"]].copy()
        har_df = har_df.rename(columns={"energy_hartree": "energy"})
        self.orbital_energies_hartree = _build_table(har_df, "Orbital energies (Hartree)")


    @classmethod
    def ui_base_schema(cls):
        base_schema = {
             "ui:order": [
                          "scf_converged","task_status","error","status",
                          'final_energy', 'files', 'energies','final_structure',
                          "structures","dipole_moment",
                          "excited_states","absorption_spectrum",
                          "vibrational_frequencies","ir_spectrum",
                          #"hyperpolarizability", was moved to elpropResult class and should stay there!
                          'bond_orders_json', 'charge', 'dipole', 'error', 'scf_energies',
                           # Orbital energies and related fields
                           "HOMO_value_eV", "LUMO_value_eV", "HOMO_LUMO_gap_eV",
                           "HOMO_value_Hartree", "LUMO_value_Hartree", "HOMO_LUMO_gap_Hartree",
                           "orbital_energies_table_eV", "orbital_energies_hartree",
                           "id"],

                         
        }

        return base_schema


async def main():
    # Example usage#
    context.initialize()
    root_dir = Path(find_project_root())
    orca_dir = root_dir / "examples" / "test1"
    orca_dir = "/home/ws/bj7610/simstack/orca/681f77f3f25308cab1d50ef1"
    # change directory to orca_dir
    os.chdir(orca_dir)
    from applications.electronic_structure.orca.pyorca import OrcaRun
    orca_run = OrcaRun("orca")
    orca_result = QMResult.from_orca_output(orca_run)
    result_dict = await orca_result.custom_model_dump()
    pprint(result_dict)
    stored_result = await context.db.save(orca_result)
    retrieved_result = await context.db.load_from_collection(QMResult, id=stored_result.id)
    result_dict = await retrieved_result.custom_model_dump()
    pprint(result_dict)

if __name__ == "__main__":
    asyncio.run(main())
