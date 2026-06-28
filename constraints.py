from enum import Enum
from typing import List, Dict, Iterator, TypeVar

from odmantic import EmbeddedModel, Model, Field

from molecular_qm_models.internal_coordinates import InternalCoordinate
from simstack.util.generic_list_mixin import GenericListMixin


class MolecularConstraintType(str, Enum):
    DISTANCE = "distance"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"
    FROZEN = "frozen"
    IMPROPER = "improper"
    HARMONIC = "harmonic"

class MolecularConstraint(EmbeddedModel):
    type: MolecularConstraintType
    atom_indices: List[int]
    parameters: Dict[str, float] = {}


    @classmethod
    def from_internal_coordinate(cls, ic: InternalCoordinate):
        atom_indices = ic.atom_indices

        if len(atom_indices) == 2:
            constraint_type = MolecularConstraintType.DISTANCE
        elif len(atom_indices) == 3:
            constraint_type = MolecularConstraintType.ANGLE
        elif len(atom_indices) == 4:
            # Could be dihedral or improper - default to dihedral
            # If more context is needed, this could be parameterized
            constraint_type = MolecularConstraintType.DIHEDRAL
        elif len(atom_indices) == 1:
            constraint_type = MolecularConstraintType.FROZEN
        else:
            raise ValueError(f"Cannot determine constraint type for {len(atom_indices)} atoms")

        return cls(type=constraint_type, atom_indices=atom_indices)


T = TypeVar("T")

class MolecularConstraintsList(Model, GenericListMixin[MolecularConstraint]):
    field_name: str = "molecular_constraints_list"
    elements: List[MolecularConstraint] = Field(default_factory=list, description="List of molecular constraints")

    def __iter__(self) -> Iterator[T]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

