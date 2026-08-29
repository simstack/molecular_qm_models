"""Tests for :class:`molecular_qm_models.molecular_geometry.Angle`.

Angles are returned in degrees (``USE_DEGREES`` defaults to ``True``).
"""

import numpy as np
import pytest

from molecular_qm_models import Angle


def test_right_angle():
    assert Angle([1, 0, 0], [0, 0, 0], [0, 1, 0]).compute() == pytest.approx(90.0)


def test_straight_angle_collinear_opposite():
    assert Angle([1, 0, 0], [0, 0, 0], [-1, 0, 0]).compute() == pytest.approx(180.0)


def test_zero_angle_same_direction():
    assert Angle([1, 0, 0], [0, 0, 0], [2, 0, 0]).compute() == pytest.approx(0.0)


def test_result_is_in_degrees():
    # 60-degree angle between the two arms.
    angle = Angle([1, 0, 0], [0, 0, 0], [0.5, np.sqrt(3) / 2, 0]).compute()
    assert angle == pytest.approx(60.0)


def test_degenerate_returns_zero_when_arm_has_zero_length():
    # p1 coincides with the vertex p2 -> undefined arm -> defined as 0.0.
    assert Angle([0, 0, 0], [0, 0, 0], [1, 0, 0]).compute() == 0.0


def test_angle_is_symmetric_in_outer_points():
    a = Angle([1, 0, 0], [0, 0, 0], [0, 1, 0]).compute()
    b = Angle([0, 1, 0], [0, 0, 0], [1, 0, 0]).compute()
    assert a == pytest.approx(b)


def test_from_molecule_matches_direct(chain):
    p1 = list(chain.atoms[0].position)
    p2 = list(chain.atoms[1].position)
    p3 = list(chain.atoms[2].position)
    expected = Angle(p1, p2, p3).compute()
    assert Angle.from_molecule(chain, 0, 1, 2) == pytest.approx(expected)
