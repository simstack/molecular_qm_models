import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from odmantic import Field, Model, Reference
from pydantic import model_validator

from molecular_qm_models.molecule import Molecule
from molecular_qm_models.qm_input import QMInput
from simstack.models import FileStack, simstack_model
from simstack.models.array_storage import ArrayStorage

GEOMETRY_HASH_DECIMALS = 6


def geometry_hash_from_molecule(molecule: Molecule) -> str:
    """SHA256 of element symbols and coordinates rounded to 6 decimal Å."""
    atoms = getattr(molecule, "atoms", None)
    if not atoms:
        raise ValueError("molecule with atoms is required for geometry_hash")
    parts = []
    for atom in atoms:
        element = getattr(atom, "element", None)
        if element is None:
            raise ValueError("atom element is required for geometry_hash")
        x = getattr(atom, "x", None)
        y = getattr(atom, "y", None)
        z = getattr(atom, "z", None)
        if x is None or y is None or z is None:
            raise ValueError("atom x, y, z coordinates are required for geometry_hash")
        parts.append(
            f"{element}:{float(x):.{GEOMETRY_HASH_DECIMALS}f}:"
            f"{float(y):.{GEOMETRY_HASH_DECIMALS}f}:{float(z):.{GEOMETRY_HASH_DECIMALS}f}"
        )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _snapshot_column_defs(field_prefix: str) -> List[Dict[str, Any]]:
    return [
        {"headerName": "date_created", "field": field_prefix + "date_created"},
        {"headerName": "task_id", "field": field_prefix + "task_id"},
        {"headerName": "smiles", "field": field_prefix + "smiles"},
        {"headerName": "formula", "field": field_prefix + "formula"},
        {"headerName": "call_path", "field": field_prefix + "call_path"},
        {"headerName": "geom_iter", "field": field_prefix + "geom_iter"},
        {"headerName": "scf_iter", "field": field_prefix + "scf_iter"},
        {"headerName": "final_structure", "field": field_prefix + "final_structure"},
        {"headerName": "energy_hartree", "field": field_prefix + "energy_hartree"},
        {"headerName": "has_forces", "field": field_prefix + "has_forces"},
        {"headerName": "|F|_rms", "field": field_prefix + "force_rms"},
        {"headerName": "geometry_hash", "field": field_prefix + "geometry_hash"},
    ]


@simstack_model
class MoleculeSnapshot(Model):
    """A geometry/wavefunction snapshot taken during a QM calculation."""

    field_name: str = "MoleculeSnapshot"
    date_created: datetime = Field(default_factory=datetime.now)
    task_id: str
    smiles: Optional[str] = None
    formula: Optional[str] = None
    call_path: Optional[str] = None
    geom_iter: int = 0
    scf_iter: int = 0
    final_structure: bool = False
    energy_hartree: Optional[float] = None
    has_forces: bool = False
    forces_hartree_bohr: Optional[ArrayStorage] = None
    geometry_hash: Optional[str] = None
    qm_input: QMInput = Reference()
    molecule: Molecule = Reference()
    wavefunction: FileStack = Reference()

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        if isinstance(data, dict) and "has_forces" not in data:
            data["has_forces"] = data.get("forces_hartree_bohr") is not None
        return data

    @model_validator(mode="after")
    def validate_forces(self):
        if not self.has_forces:
            if self.forces_hartree_bohr is not None:
                raise ValueError("forces_hartree_bohr must be None when has_forces is false")
            return self
        if self.forces_hartree_bohr is None:
            raise ValueError("forces_hartree_bohr is required when has_forces is true")
        atoms = getattr(self.molecule, "atoms", None)
        if atoms is None:
            return self
        arr = self.forces_hartree_bohr.array
        expected = (len(atoms), 3)
        if arr.ndim != 2 or tuple(arr.shape) != expected:
            raise ValueError(
                f"forces_hartree_bohr.array must have shape {expected}, got {arr.shape}"
            )
        return self

    def make_table_entries(self, **kwargs) -> Dict[str, Any]:
        field_prefix = kwargs.get("field_prefix", "")
        date_created = self.date_created.isoformat() if self.date_created else None
        force_rms = None
        if self.has_forces and self.forces_hartree_bohr is not None:
            force_rms = float(np.linalg.norm(self.forces_hartree_bohr.array))
        return {
            field_prefix + "date_created": date_created,
            field_prefix + "task_id": self.task_id,
            field_prefix + "smiles": self.smiles,
            field_prefix + "formula": self.formula,
            field_prefix + "call_path": self.call_path,
            field_prefix + "geom_iter": self.geom_iter,
            field_prefix + "scf_iter": self.scf_iter,
            field_prefix + "final_structure": self.final_structure,
            field_prefix + "energy_hartree": self.energy_hartree,
            field_prefix + "has_forces": self.has_forces,
            field_prefix + "force_rms": force_rms,
            field_prefix + "geometry_hash": self.geometry_hash,
        }

    def make_column_defs_instance(self, **kwargs) -> List[Dict[str, Any]]:
        return _snapshot_column_defs(kwargs.get("field_prefix", ""))

    @classmethod
    def make_column_defs(cls, **kwargs) -> List[Dict[str, Any]]:
        return _snapshot_column_defs(kwargs.get("field_prefix", ""))
