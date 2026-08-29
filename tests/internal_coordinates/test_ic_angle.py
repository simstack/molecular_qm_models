"""Tests for :class:`InternalAngleCoordinate` (compute and set)."""

import numpy as np
import pytest

from molecular_qm_models import Angle, InternalAngleCoordinate, InternalCoordinateType


def test_initialize_sets_type_and_indices(chain):
    ac = InternalAngleCoordinate.initialize(0, 1, 2, 90.0, 130.0, molecule=chain)
    assert ac.type == InternalCoordinateType.ANGLE
    assert ac.atom_indices == [0, 1, 2]


def test_compute_reports_current_angle(chain):
    ac = InternalAngleCoordinate.initialize(0, 1, 2, 90.0, 130.0, molecule=chain)
    ac.compute(chain)
    assert ac.real_values[0] == pytest.approx(Angle.from_molecule(chain, 0, 1, 2))


def test_set_rotates_to_target_angle(chain):
    ac = InternalAngleCoordinate.initialize(0, 1, 2, 90.0, 130.0, molecule=chain)
    ac.set(chain, 0.25)  # target actual angle = 90 + 0.25 * 40 = 100 degrees
    assert Angle.from_molecule(chain, 0, 1, 2) == pytest.approx(100.0, abs=1e-4)


def test_set_preserves_neighbouring_bond_length(chain):
    # The bond between the vertex (1) and the moving atom (2) is only rotated,
    # so its length must be preserved.
    before = np.linalg.norm(
        np.array(chain.atoms[1].position) - np.array(chain.atoms[2].position)
    )
    ac = InternalAngleCoordinate.initialize(0, 1, 2, 90.0, 130.0, molecule=chain)
    ac.set(chain, 0.5)
    after = np.linalg.norm(
        np.array(chain.atoms[1].position) - np.array(chain.atoms[2].position)
    )
    assert after == pytest.approx(before)


def test_set_then_compute_is_consistent(chain):
    ac = InternalAngleCoordinate.initialize(0, 1, 2, 90.0, 130.0, molecule=chain)
    ac.set(chain, 0.5)
    ac.compute(chain)
    assert ac.value == pytest.approx(0.5, abs=1e-4)
