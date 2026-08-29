"""Tests for :class:`InternalDihedralCoordinate` (compute and set)."""

import numpy as np
import pytest

from molecular_qm_models import (
    Dihedral,
    InternalDihedralCoordinate,
    InternalCoordinateType,
)


def test_initialize_sets_type_and_indices(chain):
    dc = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180.0, 180.0, molecule=chain)
    assert dc.type == InternalCoordinateType.DIHEDRAL
    assert dc.atom_indices == [0, 1, 2, 3]


def test_compute_reports_current_dihedral(chain):
    dc = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180.0, 180.0, molecule=chain)
    dc.compute(chain)
    assert dc.real_values[0] == pytest.approx(Dihedral.from_molecule(chain, 0, 1, 2, 3))


def test_set_changes_the_geometry(chain):
    before = Dihedral.from_molecule(chain, 0, 1, 2, 3)
    dc = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180.0, 180.0, molecule=chain)
    dc.set(chain, 0.75)
    after = Dihedral.from_molecule(chain, 0, 1, 2, 3)
    assert after != pytest.approx(before)


def test_set_preserves_the_terminal_bond_length(chain):
    before = np.linalg.norm(
        np.array(chain.atoms[2].position) - np.array(chain.atoms[3].position)
    )
    dc = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180.0, 180.0, molecule=chain)
    dc.set(chain, 0.6)
    after = np.linalg.norm(
        np.array(chain.atoms[2].position) - np.array(chain.atoms[3].position)
    )
    assert after == pytest.approx(before)


def test_set_then_compute_reflects_actual_geometry(chain):
    dc = InternalDihedralCoordinate.initialize(0, 1, 2, 3, -180.0, 180.0, molecule=chain)
    dc.set(chain, 0.6)
    dc.compute(chain)
    # compute() must report the dihedral that is actually present in the molecule.
    assert dc.real_values[0] == pytest.approx(Dihedral.from_molecule(chain, 0, 1, 2, 3))
