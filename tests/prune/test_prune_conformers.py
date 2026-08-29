"""Tests for :func:`molecular_qm_models.prune_conformers.prune_conformers`."""

import pytest

from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_models.prune_conformers import prune_conformers


# A water geometry that is clearly distinct from the default fixture geometry.
DISTINCT_SITES = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]]


def _as_list(molecules):
    ml = MoleculeList()
    for mol in molecules:
        ml.append(mol)
    return ml


def test_empty_input_returns_empty(make_conformer):
    result = list(prune_conformers(MoleculeList(), 0.1))
    assert result == []


def test_duplicates_are_pruned_keeping_lowest_energy(make_conformer):
    high = make_conformer(-1.0)
    low = make_conformer(-2.0)  # identical geometry, lower energy
    result = list(prune_conformers(_as_list([high, low]), 0.1))

    assert len(result) == 1
    assert result[0].properties["energy"] == -2.0
    assert result[0].properties["rank_id"] == 0


def test_distinct_conformers_are_all_kept(make_conformer):
    a = make_conformer(-2.0)
    b = make_conformer(-1.0, sites=DISTINCT_SITES)
    result = list(prune_conformers(_as_list([a, b]), 0.1))
    assert len(result) == 2


def test_rank_ids_follow_energy_order(make_conformer):
    a = make_conformer(-1.0, sites=DISTINCT_SITES)
    b = make_conformer(-3.0)
    result = list(prune_conformers(_as_list([a, b]), 0.1))

    by_energy = sorted(result, key=lambda m: m.properties["energy"])
    assert by_energy[0].properties["rank_id"] == 0
    assert by_energy[1].properties["rank_id"] == 1
    # The lowest energy conformer must rank first.
    assert by_energy[0].properties["energy"] == -3.0


def test_non_positive_threshold_disables_pruning(make_conformer):
    high = make_conformer(-1.0)
    low = make_conformer(-2.0)
    result = list(prune_conformers(_as_list([high, low]), 0.0))
    # Nothing is pruned and the original order is preserved.
    assert len(result) == 2
    assert [m.properties["energy"] for m in result] == [-1.0, -2.0]


def test_missing_energy_raises(make_molecule):
    mol = make_molecule(["O", "H", "H"], DISTINCT_SITES)  # no energy set
    with pytest.raises(ValueError):
        list(prune_conformers(_as_list([mol]), 0.1))
