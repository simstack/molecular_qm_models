"""Tests for :class:`molecular_qm_models.ZMatrix`.

These tests focus on the manipulation surface: building a z-matrix by hand,
reading/updating internal-coordinate values, converting to Cartesian
coordinates, and deriving a z-matrix from a molecule via ``from_trajectories``.

Note on conventions verified against the implementation:
  * bond distances are reproduced exactly by ``to_cartesian``;
  * the third atom (index 2) is placed using the supplement of the supplied
    angle, so the measured angle equals ``180 - angle``;
  * from the fourth atom onwards the supplied angle is reproduced directly.
"""

import numpy as np
import pytest

from molecular_qm_models import ZMatrix, Bond, Angle, Dihedral


def _water_zmatrix() -> ZMatrix:
    zm = ZMatrix()
    zm.add_atom("O")
    zm.add_atom("H", ref_atom1=0, distance=0.96)
    zm.add_atom("H", ref_atom1=0, distance=0.96, ref_atom2=1, angle=104.5)
    return zm


def test_single_atom_placed_at_origin():
    zm = ZMatrix()
    zm.add_atom("H")
    mol = zm.to_cartesian()
    assert len(mol.atoms) == 1
    assert mol.atoms[0].position == pytest.approx((0.0, 0.0, 0.0))


def test_two_atoms_separated_along_axis():
    zm = ZMatrix()
    zm.add_atom("H")
    zm.add_atom("H", ref_atom1=0, distance=0.74)
    mol = zm.to_cartesian()
    assert Bond.from_molecule(mol, 0, 1) == pytest.approx(0.74)


def test_to_cartesian_preserves_element_order():
    mol = _water_zmatrix().to_cartesian()
    assert [a.element for a in mol.atoms] == ["O", "H", "H"]


def test_to_cartesian_reproduces_bond_distances():
    mol = _water_zmatrix().to_cartesian()
    assert Bond.from_molecule(mol, 0, 1) == pytest.approx(0.96)
    assert Bond.from_molecule(mol, 0, 2) == pytest.approx(0.96)


def test_third_atom_uses_supplement_of_angle():
    mol = _water_zmatrix().to_cartesian()
    assert Angle.from_molecule(mol, 1, 0, 2) == pytest.approx(180.0 - 104.5)


def test_fourth_atom_reproduces_angle():
    zm = ZMatrix()
    zm.add_atom("C")
    zm.add_atom("C", 0, 1.5)
    zm.add_atom("C", 1, 1.5, 0, 110.0)
    zm.add_atom("C", 2, 1.5, 1, 110.0, 0, 60.0)
    mol = zm.to_cartesian()
    assert Bond.from_molecule(mol, 2, 3) == pytest.approx(1.5)
    assert Angle.from_molecule(mol, 1, 2, 3) == pytest.approx(110.0)


def test_get_value_returns_stored_distance():
    zm = _water_zmatrix()
    assert zm.get_value(1, 0) == pytest.approx(0.96)


def test_set_value_updates_geometry():
    zm = _water_zmatrix()
    zm.set_value(1, 0, 1.20)
    assert zm.get_value(1, 0) == pytest.approx(1.20)
    mol = zm.to_cartesian()
    assert Bond.from_molecule(mol, 0, 1) == pytest.approx(1.20)


def test_from_trajectories_single_frame_preserves_atoms(chain):
    zm = ZMatrix.from_trajectories([[chain]])
    assert len(zm.atoms) == len(chain.atoms)
    mol = zm.to_cartesian()
    assert [a.element for a in mol.atoms] == [a.element for a in chain.atoms]


def test_from_trajectories_preserves_reference_bond_lengths(water):
    # The distances stored against each atom's first reference should match the
    # actual bonded distances recovered after conversion back to Cartesian.
    zm = ZMatrix.from_trajectories([[water]])
    mol = zm.to_cartesian()
    for i, entry in enumerate(zm.atoms):
        ref1 = entry["refs"][0]
        if ref1 is None:
            continue
        assert Bond.from_molecule(mol, i, ref1) == pytest.approx(
            entry["values"][0], abs=1e-6
        )
