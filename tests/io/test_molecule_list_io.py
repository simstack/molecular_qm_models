"""Tests for multi-structure :class:`MoleculeList` file I/O."""

import pytest

from molecular_qm_models import Molecule, MoleculeList


WATER_XYZ = "3\nwater\nO 0.0 0.0 0.0\nH 0.758 0.586 0.0\nH -0.758 0.586 0.0\n"


def _list_of(n: int) -> MoleculeList:
    ml = MoleculeList()
    for _ in range(n):
        ml.append(Molecule.from_xyz(WATER_XYZ))
    return ml


def test_write_and_read_multi_xyz_roundtrip(tmp_path):
    ml = _list_of(3)
    path = tmp_path / "many.xyz"
    ml.to_file(str(path))

    loaded = MoleculeList.from_file(str(path))
    molecules = list(loaded)
    assert len(molecules) == 3
    for mol in molecules:
        assert [a.element for a in mol.atoms] == ["O", "H", "H"]


def test_from_file_start_offset(tmp_path):
    ml = _list_of(3)
    path = tmp_path / "many.xyz"
    ml.to_file(str(path))

    loaded = MoleculeList.from_file(str(path), start=1)
    assert len(list(loaded)) == 2


def test_from_file_number_limit(tmp_path):
    ml = _list_of(3)
    path = tmp_path / "many.xyz"
    ml.to_file(str(path))

    loaded = MoleculeList.from_file(str(path), start=1, number=1)
    assert len(list(loaded)) == 1


def test_from_file_sdf_roundtrip(tmp_path):
    ml = _list_of(2)
    path = tmp_path / "many.sdf"
    ml.to_file(str(path))

    loaded = MoleculeList.from_file(str(path))
    molecules = list(loaded)
    assert len(molecules) == 2
    for mol in molecules:
        assert len(mol.atoms) == 3
