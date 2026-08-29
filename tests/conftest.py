"""Shared fixtures for the ``molecular_qm_models`` test suite.

The fixtures here build small, well understood molecules that are reused across
the ``molecule``, ``geometry``, ``internal_coordinates``, ``zmatrix``,
``alignment`` and ``prune`` test packages.  All molecule fixtures return a fresh
object on every use so tests are free to mutate them in place.
"""

from typing import Callable, List, Sequence

import pytest

from molecular_qm_models import Molecule


# A planar water molecule (O, H, H).  Handy because it is small and achiral.
WATER_ELEMENTS: List[str] = ["O", "H", "H"]
WATER_SITES: List[List[float]] = [
    [0.0, 0.0, 0.0],
    [0.758, 0.586, 0.0],
    [-0.758, 0.586, 0.0],
]

# A four atom chain (C-C-O-H) whose atoms are covalently bonded in sequence.
# This gives genuine bond/angle/dihedral values and a linear connectivity graph
# which is convenient for internal-coordinate and z-matrix tests.
CHAIN_ELEMENTS: List[str] = ["C", "C", "O", "H"]
CHAIN_SITES: List[List[float]] = [
    [0.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [2.0, 1.2, 0.0],
    [3.0, 1.2, 0.5],
]


@pytest.fixture
def make_molecule() -> Callable[[Sequence[str], Sequence[Sequence[float]]], Molecule]:
    """Return a factory that builds a fresh :class:`Molecule` from sites."""

    def _make(elements: Sequence[str], sites: Sequence[Sequence[float]]) -> Molecule:
        return Molecule.from_sites(list(elements), [list(s) for s in sites])

    return _make


@pytest.fixture
def water(make_molecule) -> Molecule:
    """A fresh, planar water molecule."""
    return make_molecule(WATER_ELEMENTS, WATER_SITES)


@pytest.fixture
def chain(make_molecule) -> Molecule:
    """A fresh C-C-O-H chain with sequential covalent connectivity."""
    return make_molecule(CHAIN_ELEMENTS, CHAIN_SITES)


@pytest.fixture
def make_conformer(make_molecule) -> Callable[..., Molecule]:
    """Return a factory that builds a water conformer tagged with an energy."""

    def _make(energy: float, sites: Sequence[Sequence[float]] = None) -> Molecule:
        mol = make_molecule(WATER_ELEMENTS, sites if sites is not None else WATER_SITES)
        mol.properties["energy"] = energy
        return mol

    return _make
