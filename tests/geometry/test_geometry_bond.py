"""Tests for :class:`molecular_qm_models.molecular_geometry.Bond`."""

import math

import numpy as np
import pytest

from molecular_qm_models import Bond


def test_unit_distance_along_axis():
    assert Bond([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).compute() == pytest.approx(1.0)


def test_zero_distance_for_coincident_points():
    assert Bond([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]).compute() == pytest.approx(0.0)


def test_general_distance_matches_numpy():
    p1 = [1.0, -2.0, 0.5]
    p2 = [-3.0, 4.0, 2.5]
    expected = float(np.linalg.norm(np.array(p1) - np.array(p2)))
    assert Bond(p1, p2).compute() == pytest.approx(expected)


def test_symmetry_of_distance():
    p1 = [0.1, 0.2, 0.3]
    p2 = [1.1, -0.7, 2.0]
    assert Bond(p1, p2).compute() == pytest.approx(Bond(p2, p1).compute())


def test_accepts_numpy_arrays():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    assert Bond(p1, p2).compute() == pytest.approx(5.0)


def test_from_molecule_matches_direct_computation(water):
    a0 = water.atoms[0]
    a1 = water.atoms[1]
    expected = math.dist(a0.position, a1.position)
    assert Bond.from_molecule(water, 0, 1) == pytest.approx(expected)


def test_from_molecule_symmetric(chain):
    assert Bond.from_molecule(chain, 0, 1) == pytest.approx(
        Bond.from_molecule(chain, 1, 0)
    )
