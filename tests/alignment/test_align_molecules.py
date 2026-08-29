"""Tests for :func:`molecular_qm_models.align_molecules.align_molecules`."""

import numpy as np
import pytest

from molecular_qm_models import Molecule
from molecular_qm_models.align_molecules import align_molecules


def _coords(mol: Molecule) -> np.ndarray:
    return np.array([a.position for a in mol.atoms], dtype=float)


def _rotate_translate(mol: Molecule, theta: float, shift) -> Molecule:
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = Molecule.from_molecule(mol)
    for atom in moved.atoms:
        atom.position = (rot @ np.array(atom.position) + np.array(shift)).tolist()
    return moved


def test_identical_molecules_have_zero_rmsd(water):
    _, _, rmsd = align_molecules(water, Molecule.from_molecule(water))
    assert rmsd == pytest.approx(0.0, abs=1e-9)


def test_rotated_translated_copy_aligns_with_zero_rmsd(water):
    mobile = _rotate_translate(water, 0.9, [3.0, -2.0, 1.0])
    _, _, rmsd = align_molecules(water, mobile)
    assert rmsd == pytest.approx(0.0, abs=1e-6)


def test_aligned_molecules_are_centred_on_origin(water):
    mobile = _rotate_translate(water, 0.5, [1.0, 1.0, 1.0])
    aligned_ref, aligned_mobile, _ = align_molecules(water, mobile)
    assert np.allclose(_coords(aligned_ref).mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(_coords(aligned_mobile).mean(axis=0), 0.0, atol=1e-9)


def test_inputs_are_not_modified(water):
    original = _coords(water).copy()
    mobile = _rotate_translate(water, 0.3, [2.0, 0.0, 0.0])
    mobile_before = _coords(mobile).copy()

    align_molecules(water, mobile)

    assert np.allclose(_coords(water), original)
    assert np.allclose(_coords(mobile), mobile_before)


def test_returns_independent_copies(water):
    aligned_ref, aligned_mobile, _ = align_molecules(
        water, Molecule.from_molecule(water)
    )
    assert aligned_ref is not water
    assert aligned_mobile is not water


def test_mismatched_atom_counts_raise():
    a = Molecule.from_sites(["O", "H"], [[0, 0, 0], [1, 0, 0]])
    b = Molecule.from_sites(["O"], [[0, 0, 0]])
    with pytest.raises(ValueError):
        align_molecules(a, b)


def test_rmsd_is_positive_for_genuinely_different_shapes():
    a = Molecule.from_sites(["H", "H", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    b = Molecule.from_sites(["H", "H", "H"], [[0, 0, 0], [2, 0, 0], [0, 0.5, 0]])
    _, _, rmsd = align_molecules(a, b)
    assert rmsd > 0.0
