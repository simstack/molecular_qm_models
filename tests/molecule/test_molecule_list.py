"""Tests for :class:`molecular_qm_models.MoleculeList` container behaviour."""

from molecular_qm_models import Molecule, MoleculeList


def test_empty_list_iterates_to_nothing():
    assert list(MoleculeList()) == []


def test_append_and_iterate(water):
    ml = MoleculeList()
    ml.append(water)
    ml.append(Molecule.from_molecule(water))

    molecules = list(ml)
    assert len(molecules) == 2
    for mol in molecules:
        assert isinstance(mol, Molecule)
        assert len(mol.atoms) == len(water.atoms)


def test_iteration_preserves_order(make_molecule):
    first = make_molecule(["O"], [[0.0, 0.0, 0.0]])
    second = make_molecule(["N"], [[1.0, 0.0, 0.0]])

    ml = MoleculeList()
    ml.append(first)
    ml.append(second)

    elements = [mol.atoms[0].element for mol in ml]
    assert elements == ["O", "N"]
