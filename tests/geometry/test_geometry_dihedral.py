"""Tests for :class:`molecular_qm_models.molecular_geometry.Dihedral`.

Dihedrals are returned in degrees.  For the reference geometry used below
(``p1=[1,0,0]``, ``p2=[0,0,0]``, ``p3=[0,0,1]``, ``p4=[cos t, sin t, 1]``) the
implementation yields ``-t`` degrees.
"""

import numpy as np
import pytest

from molecular_qm_models import Dihedral


P1 = [1.0, 0.0, 0.0]
P2 = [0.0, 0.0, 0.0]
P3 = [0.0, 0.0, 1.0]


def _p4(t_deg: float):
    t = np.radians(t_deg)
    return [float(np.cos(t)), float(np.sin(t)), 1.0]


@pytest.mark.parametrize(
    "t_deg, expected",
    [
        (0.0, 0.0),
        (45.0, -45.0),
        (90.0, -90.0),
        (135.0, -135.0),
        (180.0, -180.0),
        (-90.0, 90.0),
    ],
)
def test_dihedral_sign_and_magnitude(t_deg, expected):
    assert Dihedral(P1, P2, P3, _p4(t_deg)).compute() == pytest.approx(expected)


def test_planar_cis_is_zero():
    assert Dihedral(P1, P2, P3, _p4(0.0)).compute() == pytest.approx(0.0)


def test_reversed_atom_order_gives_same_dihedral():
    p4 = [0.0, 1.0, 1.0]
    forward = Dihedral(P1, P2, P3, p4).compute()
    reverse = Dihedral(p4, P3, P2, P1).compute()
    assert forward == pytest.approx(reverse)


def test_degenerate_returns_zero_for_collinear_points():
    # All points along a single line -> zero cross products -> defined as 0.0.
    assert Dihedral([0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]).compute() == 0.0


def test_degenerate_returns_zero_for_coincident_points():
    assert Dihedral([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]).compute() == 0.0


def test_from_molecule_matches_direct(chain):
    coords = [list(chain.atoms[i].position) for i in range(4)]
    expected = Dihedral(*coords).compute()
    assert Dihedral.from_molecule(chain, 0, 1, 2, 3) == pytest.approx(expected)
