"""Tests for :meth:`molecular_qm_models.Molecule.complex_hash`.

``complex_hash`` produces an order-, translation- and rotation-invariant
geometric fingerprint of a molecule.
"""

import numpy as np

from molecular_qm_models import Atom, Molecule


def _translated(mol: Molecule, shift) -> Molecule:
    clone = Molecule.from_molecule(mol)
    for atom in clone.atoms:
        atom.x += shift[0]
        atom.y += shift[1]
        atom.z += shift[2]
    return clone


def _rotated_z(mol: Molecule, theta: float) -> Molecule:
    clone = Molecule.from_molecule(mol)
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    for atom in clone.atoms:
        atom.x, atom.y, atom.z = (rot @ np.array(atom.position)).tolist()
    return clone


def _reordered(mol: Molecule, order) -> Molecule:
    return Molecule.from_atoms([Atom.from_atom(mol.atoms[i]) for i in order])


def test_hash_is_deterministic(water):
    assert water.complex_hash() == water.complex_hash()


def test_hash_is_a_hex_string(water):
    digest = water.complex_hash()
    assert isinstance(digest, str)
    assert len(digest) == 32
    int(digest, 16)  # must be valid hexadecimal


def test_hash_is_translation_invariant(water):
    assert _translated(water, [5.0, -3.0, 2.0]).complex_hash() == water.complex_hash()


def test_hash_is_rotation_invariant(water):
    assert _rotated_z(water, 0.7).complex_hash() == water.complex_hash()


def test_hash_is_order_invariant(water):
    assert _reordered(water, [2, 0, 1]).complex_hash() == water.complex_hash()


def test_hash_distinguishes_different_geometries(water):
    distorted = Molecule.from_molecule(water)
    distorted.atoms[1].x += 0.5
    assert distorted.complex_hash() != water.complex_hash()


def test_precision_changes_hash(water):
    assert water.complex_hash(precision=1) != water.complex_hash(precision=6)


def test_preserve_chirality_accepts_flag_and_returns_valid_hash(water):
    digest = water.complex_hash(preserve_chirality=True)
    assert isinstance(digest, str)
    assert len(digest) == 32
