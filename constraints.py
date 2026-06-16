from enum import Enum
from typing import List

from odmantic import EmbeddedModel

from molecular_qm_models.internal_coordinates import InternalCoordinate


class MolecularConstraintType(str, Enum):
    DISTANCE = "distance"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"
    FROZEN = "frozen"
    IMPROPER = "improper"

class MolecularConstraint(EmbeddedModel):
    type: MolecularConstraintType
    atom_indices: List[int]


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
