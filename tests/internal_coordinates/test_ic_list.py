"""Tests for :class:`InternalCoordinatesList` builders.

The concrete coordinates are exposed through the ``.elements`` attribute.
"""

from molecular_qm_models import (
    InternalCoordinatesList,
    InternalBondCoordinate,
    InternalAngleCoordinate,
    InternalDihedralCoordinate,
    InternalCoordinateType,
)


def _with_selections(chain, selections):
    chain.properties["selections"] = selections
    return InternalCoordinatesList.from_molecule(chain)


def test_from_molecule_builds_bond_coordinate(chain):
    icl = _with_selections(chain, [{"type": "bond", "atoms": [0, 1]}])
    assert len(icl.elements) == 1
    coord = icl.elements[0]
    assert isinstance(coord, InternalBondCoordinate)
    assert coord.type == InternalCoordinateType.BOND
    assert coord.atom_indices == [0, 1]
    # Range brackets the current bond length.
    assert coord.min_values[0] < 1.5 < coord.max_values[0]


def test_from_molecule_builds_angle_coordinate(chain):
    icl = _with_selections(chain, [{"type": "angle", "atoms": [0, 1, 2]}])
    coord = icl.elements[0]
    assert isinstance(coord, InternalAngleCoordinate)
    assert coord.atom_indices == [0, 1, 2]


def test_from_molecule_builds_dihedral_coordinate(chain):
    icl = _with_selections(chain, [{"type": "dihedral", "atoms": [0, 1, 2, 3]}])
    coord = icl.elements[0]
    assert isinstance(coord, InternalDihedralCoordinate)
    assert coord.atom_indices == [0, 1, 2, 3]


def test_from_molecule_builds_multiple_coordinates(chain):
    icl = _with_selections(
        chain,
        [{"type": "bond", "atoms": [0, 1]}, {"type": "angle", "atoms": [0, 1, 2]}],
    )
    types = [type(c) for c in icl.elements]
    assert types == [InternalBondCoordinate, InternalAngleCoordinate]


def test_from_states_defines_range_from_two_geometries(chain, make_molecule):
    state1 = chain
    state1.properties["selections"] = [{"type": "dihedral", "atoms": [0, 1, 2, 3]}]
    state2 = make_molecule(
        ["C", "C", "O", "H"],
        [[0, 0, 0], [1.5, 0, 0], [2.0, 1.2, 0], [3.0, 1.2, -0.5]],
    )

    icl = InternalCoordinatesList.from_states(state1, state2)
    coord = icl.elements[0]
    assert isinstance(coord, InternalDihedralCoordinate)
    assert coord.atom_indices == [0, 1, 2, 3]
    # The range must be non-degenerate (min strictly below max).
    assert coord.min_values[0] < coord.max_values[0]


def test_internal_coordinates_list_iteration_and_len(chain):
    icl = _with_selections(
        chain,
        [{"type": "bond", "atoms": [0, 1]}, {"type": "angle", "atoms": [0, 1, 2]}],
    )
    assert len(icl) == 2
    elements = [coord for coord in icl]
    assert len(elements) == 2
    assert isinstance(icl[0], InternalBondCoordinate)
    assert isinstance(icl[1], InternalAngleCoordinate)


def test_internal_fragment_coordinate_uses_internal_coordinates_list(chain, make_molecule):
    from molecular_qm_models.internal_coordinates import InternalFragmentCoordinate

    mol1 = chain
    mol2 = make_molecule(
        ["C", "C", "O", "H"],
        [[0, 0, 0], [1.6, 0, 0], [2.1, 1.3, 0], [3.1, 1.3, -0.6]],
    )
    frag = InternalFragmentCoordinate.initialize([0, 1, 2], mol1, mol2)
    assert isinstance(frag.bonds, InternalCoordinatesList)
    assert isinstance(frag.angles, InternalCoordinatesList)
    assert isinstance(frag.dihedrals, InternalCoordinatesList)
    assert len(frag.bonds) > 0
    assert len(frag.angles) > 0

    # Test default initialization
    default_frag = InternalFragmentCoordinate(
        atom_indices=[0, 1], min_values=[0.0], max_values=[1.0]
    )
    assert isinstance(default_frag.bonds, InternalCoordinatesList)
    assert isinstance(default_frag.angles, InternalCoordinatesList)
    assert isinstance(default_frag.dihedrals, InternalCoordinatesList)
