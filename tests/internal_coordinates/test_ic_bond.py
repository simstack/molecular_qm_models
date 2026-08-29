"""Tests for :class:`InternalBondCoordinate` (compute and set)."""

import numpy as np
import pytest

from molecular_qm_models import Bond, InternalBondCoordinate, InternalCoordinateType


def test_initialize_sets_type_and_indices(chain):
    bc = InternalBondCoordinate.initialize(0, 1, 1.0, 2.0, molecule=chain)
    assert bc.type == InternalCoordinateType.BOND
    assert bc.atom_indices == [0, 1]
    assert bc.min_values == [1.0]
    assert bc.max_values == [2.0]


def test_initialize_populates_moving_atoms(chain):
    bc = InternalBondCoordinate.initialize(0, 1, 1.0, 2.0, molecule=chain)
    # Everything reachable from atom 1 without passing through atom 0.
    assert set(bc.moving_atoms) == {1, 2, 3}


def test_compute_reports_current_length_and_normalised_value(chain):
    bc = InternalBondCoordinate.initialize(0, 1, 1.0, 2.0, molecule=chain)
    bc.compute(chain)
    assert bc.real_values[0] == pytest.approx(Bond.from_molecule(chain, 0, 1))
    assert bc.real_values[0] == pytest.approx(1.5)
    # (1.5 - 1.0) / (2.0 - 1.0) = 0.5
    assert bc.value == pytest.approx(0.5)


def test_set_changes_bond_length_to_target(chain):
    bc = InternalBondCoordinate.initialize(0, 1, 1.0, 2.0, molecule=chain)
    bc.set(chain, 0.75)  # target actual length = 1.75
    assert Bond.from_molecule(chain, 0, 1) == pytest.approx(1.75)


def test_set_leaves_anchor_atom_fixed(chain):
    anchor_before = np.array(chain.atoms[0].position)
    bc = InternalBondCoordinate.initialize(0, 1, 1.0, 2.0, molecule=chain)
    bc.set(chain, 0.9)
    anchor_after = np.array(chain.atoms[0].position)
    assert np.allclose(anchor_before, anchor_after)


def test_set_then_compute_is_consistent(chain):
    bc = InternalBondCoordinate.initialize(0, 1, 1.0, 2.0, molecule=chain)
    bc.set(chain, 0.25)
    bc.compute(chain)
    assert bc.value == pytest.approx(0.25)
