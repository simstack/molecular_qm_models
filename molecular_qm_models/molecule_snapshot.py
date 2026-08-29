from datetime import datetime
from typing import Any, Dict, List, Optional

from odmantic import Field, Model, Reference
from pydantic import model_validator

from molecular_qm_models.molecule import Molecule
from molecular_qm_models.qm_input import QMInput
from simstack.models import FileStack, simstack_model


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
    qm_input: QMInput = Reference()
    molecule: Molecule = Reference()
    wavefunction: FileStack = Reference()

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    def make_table_entries(self, **kwargs) -> Dict[str, Any]:
        field_prefix = kwargs.get("field_prefix", "")
        date_created = self.date_created.isoformat() if self.date_created else None
        return {
            field_prefix + "date_created": date_created,
            field_prefix + "task_id": self.task_id,
            field_prefix + "smiles": self.smiles,
            field_prefix + "formula": self.formula,
            field_prefix + "call_path": self.call_path,
            field_prefix + "geom_iter": self.geom_iter,
            field_prefix + "scf_iter": self.scf_iter,
            field_prefix + "final_structure": self.final_structure,
        }

    def make_column_defs_instance(self, **kwargs) -> List[Dict[str, Any]]:
        field_prefix = kwargs.get("field_prefix", "")
        return [
            {"headerName": "date_created", "field": field_prefix + "date_created"},
            {"headerName": "task_id", "field": field_prefix + "task_id"},
            {"headerName": "smiles", "field": field_prefix + "smiles"},
            {"headerName": "formula", "field": field_prefix + "formula"},
            {"headerName": "call_path", "field": field_prefix + "call_path"},
            {"headerName": "geom_iter", "field": field_prefix + "geom_iter"},
            {"headerName": "scf_iter", "field": field_prefix + "scf_iter"},
            {"headerName": "final_structure", "field": field_prefix + "final_structure"},
        ]

    @classmethod
    def make_column_defs(cls, **kwargs) -> List[Dict[str, Any]]:
        field_prefix = kwargs.get("field_prefix", "")
        return [
            {"headerName": "date_created", "field": field_prefix + "date_created"},
            {"headerName": "task_id", "field": field_prefix + "task_id"},
            {"headerName": "smiles", "field": field_prefix + "smiles"},
            {"headerName": "formula", "field": field_prefix + "formula"},
            {"headerName": "call_path", "field": field_prefix + "call_path"},
            {"headerName": "geom_iter", "field": field_prefix + "geom_iter"},
            {"headerName": "scf_iter", "field": field_prefix + "scf_iter"},
            {"headerName": "final_structure", "field": field_prefix + "final_structure"},
        ]
