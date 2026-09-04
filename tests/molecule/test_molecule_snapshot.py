"""Tests for MoleculeSnapshot force fields and geometry hashing."""

import numpy as np
import pytest

from molecular_qm_models import (
    BasisSet,
    Functional,
    Molecule,
    MoleculeSnapshot,
    QMInput,
    geometry_hash_from_molecule,
)
from simstack.models import FileStack
from simstack.models.array_storage import ArrayStorage


def _qm_input(molecule: Molecule) -> QMInput:
    return QMInput(
        molecule=molecule,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
    )


def _wavefunction() -> FileStack:
    return FileStack.from_string("wfn", "snapshot.wfn.npy")


def _forces_storage(values) -> ArrayStorage:
    storage = ArrayStorage(name="forces_hartree_bohr")
    storage.array = np.asarray(values, dtype=float)
    return storage


def test_geometry_hash_is_stable(water):
    first = geometry_hash_from_molecule(water)
    second = geometry_hash_from_molecule(water)
    assert first == second
    assert len(first) == 64


def test_geometry_hash_changes_with_coordinates(water, make_molecule):
    moved = make_molecule(
        ["O", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.758, 0.586, 0.0],
            [-0.758, 0.586, 0.1],
        ],
    )
    assert geometry_hash_from_molecule(water) != geometry_hash_from_molecule(moved)


def test_geometry_hash_requires_atoms():
    with pytest.raises(ValueError, match="molecule with atoms is required"):
        geometry_hash_from_molecule(Molecule())


def test_snapshot_without_forces_loads_legacy_defaults(water):
    snapshot = MoleculeSnapshot(
        task_id="task-1",
        molecule=water,
        qm_input=_qm_input(water),
        wavefunction=_wavefunction(),
    )
    assert snapshot.has_forces is False
    assert snapshot.forces_hartree_bohr is None
    assert snapshot.energy_hartree is None
    table = snapshot.make_table_entries()
    assert table["has_forces"] is False
    assert table["force_rms"] is None
    columns = {col["field"] for col in snapshot.make_column_defs_instance()}
    assert "energy_hartree" in columns
    assert "has_forces" in columns
    assert "force_rms" in columns
    assert "geometry_hash" in columns


def test_snapshot_has_forces_false_rejects_storage(water):
    with pytest.raises(ValueError, match="must be None when has_forces is false"):
        MoleculeSnapshot(
            task_id="task-1",
            molecule=water,
            qm_input=_qm_input(water),
            wavefunction=_wavefunction(),
            has_forces=False,
            forces_hartree_bohr=_forces_storage([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
        )


def test_snapshot_has_forces_true_requires_storage(water):
    with pytest.raises(ValueError, match="required when has_forces is true"):
        MoleculeSnapshot(
            task_id="task-1",
            molecule=water,
            qm_input=_qm_input(water),
            wavefunction=_wavefunction(),
            has_forces=True,
        )


def test_snapshot_force_shape_mismatch_raises(water):
    with pytest.raises(ValueError, match="must have shape"):
        MoleculeSnapshot(
            task_id="task-1",
            molecule=water,
            qm_input=_qm_input(water),
            wavefunction=_wavefunction(),
            has_forces=True,
            forces_hartree_bohr=_forces_storage([[0.1, 0.0, 0.0]]),
        )


def test_snapshot_valid_forces(water):
    forces = np.array(
        [[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3]],
        dtype=float,
    )
    snapshot = MoleculeSnapshot(
        task_id="task-1",
        molecule=water,
        qm_input=_qm_input(water),
        wavefunction=_wavefunction(),
        energy_hartree=-76.4,
        has_forces=True,
        forces_hartree_bohr=_forces_storage(forces),
        geometry_hash=geometry_hash_from_molecule(water),
    )
    np.testing.assert_allclose(snapshot.forces_hartree_bohr.array, forces)
    table = snapshot.make_table_entries()
    assert table["energy_hartree"] == pytest.approx(-76.4)
    assert table["has_forces"] is True
    assert table["force_rms"] == pytest.approx(float(np.linalg.norm(forces)))
    assert table["geometry_hash"] == geometry_hash_from_molecule(water)
