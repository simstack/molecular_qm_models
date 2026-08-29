"""Tests for :class:`molecular_qm_models.Molecule` construction and manipulation."""

import pytest

from molecular_qm_models import Atom, Molecule


def test_from_sites_builds_expected_atoms():
    mol = Molecule.from_sites(["O", "H"], [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    assert [a.element for a in mol.atoms] == ["O", "H"]
    assert mol.atoms[1].position == (0.9, 0.0, 0.0)


def test_from_sites_length_mismatch_raises():
    with pytest.raises(ValueError):
        Molecule.from_sites(["O", "H"], [[0.0, 0.0, 0.0]])


def test_from_atoms():
    atoms = [Atom.from_coords("C", [0, 0, 0]), Atom.from_coords("H", [1, 0, 0])]
    mol = Molecule.from_atoms(atoms)
    assert len(mol.atoms) == 2
    assert [a.element for a in mol.atoms] == ["C", "H"]


def test_from_molecule_is_deep_copy(water):
    clone = Molecule.from_molecule(water)
    assert clone is not water
    assert clone.atoms[0] is not water.atoms[0]
    assert [a.element for a in clone.atoms] == [a.element for a in water.atoms]

    # Mutating the clone must not affect the original.
    clone.atoms[0].x += 10.0
    assert clone.atoms[0].x != water.atoms[0].x


def test_add_atom_mutates_in_place(water):
    n_before = len(water.atoms)
    water.add_atom(Atom.from_coords("H", [5.0, 5.0, 5.0]))
    assert len(water.atoms) == n_before + 1
    assert water.atoms[-1].position == (5.0, 5.0, 5.0)


def test_len_returns_atom_count(water):
    assert len(water) == len(water.atoms) == 3


def test_getitem_returns_atom(water):
    assert water[0] is water.atoms[0]


def test_iter_yields_atoms(water):
    assert list(iter(water)) == list(water.atoms)


def test_default_charge_and_spin(water):
    assert water.charge == 0
    assert water.spin_multiplicity == 1


def test_charge_and_spin_read_from_properties(water):
    water.properties["charge"] = -1
    water.properties["spin_multiplicity"] = 2
    assert water.charge == -1
    assert water.spin_multiplicity == 2
