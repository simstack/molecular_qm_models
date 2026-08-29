"""Tests for single-:class:`Molecule` file I/O (XYZ parsing and writing)."""

import pytest

from molecular_qm_models import Molecule


WATER_XYZ = "3\nwater\nO 0.0 0.0 0.0\nH 0.758 0.586 0.0\nH -0.758 0.586 0.0\n"


def test_from_xyz_parses_elements_and_coordinates():
    mol = Molecule.from_xyz(WATER_XYZ)
    assert [a.element for a in mol.atoms] == ["O", "H", "H"]
    assert mol.atoms[0].position == (0.0, 0.0, 0.0)
    assert mol.atoms[1].position == pytest.approx((0.758, 0.586, 0.0))


def test_to_file_then_from_file_roundtrip(tmp_path):
    mol = Molecule.from_xyz(WATER_XYZ)
    path = tmp_path / "mol.xyz"
    mol.to_file(str(path))

    assert path.exists()
    loaded = Molecule.from_file(str(path))
    assert [a.element for a in loaded.atoms] == [a.element for a in mol.atoms]
    for original, restored in zip(mol.atoms, loaded.atoms):
        assert restored.position == pytest.approx(original.position)


def test_from_file_missing_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist.xyz"
    with pytest.raises(FileNotFoundError):
        Molecule.from_file(str(missing))


def test_from_file_unsupported_extension_raises(tmp_path):
    path = tmp_path / "mol.unsupported"
    path.write_text("garbage")
    with pytest.raises(ValueError):
        Molecule.from_file(str(path))


def test_to_file_unsupported_extension_raises(tmp_path):
    mol = Molecule.from_xyz(WATER_XYZ)
    path = tmp_path / "mol.unsupported"
    with pytest.raises(ValueError):
        mol.to_file(str(path))
