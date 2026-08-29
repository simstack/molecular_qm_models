"""Tests for helpers on :class:`InternalCoordinateBase`.

Covers the graph utilities (``_get_adjacency`` / ``get_moving_atoms``), the
normalised-value interpolation (``get_actual_value``) and the static geometry
delegates (``get_distance`` / ``get_angle`` / ``get_dihedral``).
"""

import pytest

from molecular_qm_models import Bond, Angle, Dihedral, InternalCoordinateType
from molecular_qm_models.internal_coordinates import InternalCoordinateBase


def _base():
    return InternalCoordinateBase(
        type=InternalCoordinateType.BOND,
        atom_indices=[0, 1],
        min_values=[1.0],
        max_values=[2.0],
    )


def test_get_adjacency_for_linear_chain(chain):
    adj = InternalCoordinateBase._get_adjacency(chain)
    assert adj == [[1], [0, 2], [1, 3], [2]]


def test_get_moving_atoms_one_side_of_bond(chain):
    moving = InternalCoordinateBase.get_moving_atoms(chain, (1, 2))
    assert set(moving) == {2, 3}


def test_get_moving_atoms_other_side_of_bond(chain):
    moving = InternalCoordinateBase.get_moving_atoms(chain, (2, 1))
    assert set(moving) == {0, 1}


def test_get_moving_atoms_respects_collision_set(chain):
    moving = InternalCoordinateBase.get_moving_atoms(chain, (1, 2), collision_set=[3])
    assert set(moving) == {2}


@pytest.mark.parametrize(
    "norm, expected",
    [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0), (0.25, 1.25)],
)
def test_get_actual_value_linear_interpolation(norm, expected):
    assert _base().get_actual_value(norm) == pytest.approx(expected)


def test_static_get_distance_matches_bond():
    p1, p2 = [0.0, 0.0, 0.0], [0.0, 3.0, 4.0]
    assert InternalCoordinateBase.get_distance(p1, p2) == pytest.approx(
        Bond(p1, p2).compute()
    )


def test_static_get_angle_matches_angle():
    p1, p2, p3 = [1, 0, 0], [0, 0, 0], [0, 1, 0]
    assert InternalCoordinateBase.get_angle(p1, p2, p3) == pytest.approx(
        Angle(p1, p2, p3).compute()
    )


def test_static_get_dihedral_matches_dihedral():
    p1, p2, p3, p4 = [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]
    assert InternalCoordinateBase.get_dihedral(p1, p2, p3, p4) == pytest.approx(
        Dihedral(p1, p2, p3, p4).compute()
    )
