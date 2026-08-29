"""Tests for :class:`molecular_qm_models.Atom`."""

import math

import pytest

from molecular_qm_models import Atom


def test_construct_with_keyword_fields():
    atom = Atom(element="H", x=1.0, y=2.0, z=3.0)
    assert atom.element == "H"
    assert (atom.x, atom.y, atom.z) == (1.0, 2.0, 3.0)


def test_from_coords():
    atom = Atom.from_coords("O", [1.0, -2.0, 0.5])
    assert atom.element == "O"
    assert atom.position == (1.0, -2.0, 0.5)


def test_from_coords_too_few_values_raises():
    with pytest.raises(IndexError):
        Atom.from_coords("O", [0.0, 1.0])


def test_from_atom_copies_values_and_properties():
    original = Atom(element="C", x=1.0, y=2.0, z=3.0)
    original.properties["tag"] = "a"
    clone = Atom.from_atom(original)

    assert clone is not original
    assert clone.element == "C"
    assert clone.position == (1.0, 2.0, 3.0)

    # The properties dict must be an independent copy.
    clone.properties["tag"] = "b"
    assert original.properties["tag"] == "a"


def test_iter_yields_coordinates_in_order():
    atom = Atom(element="H", x=1.0, y=2.0, z=3.0)
    assert list(atom) == [1.0, 2.0, 3.0]
    x, y, z = atom
    assert (x, y, z) == (1.0, 2.0, 3.0)


def test_species_property_aliases_element():
    assert Atom(element="Na", x=0, y=0, z=0).species == "Na"


def test_position_getter_returns_tuple():
    atom = Atom(element="H", x=1.0, y=2.0, z=3.0)
    assert atom.position == (1.0, 2.0, 3.0)


def test_position_setter_updates_coordinates():
    atom = Atom(element="H", x=0.0, y=0.0, z=0.0)
    atom.position = [4.0, 5.0, 6.0]
    assert (atom.x, atom.y, atom.z) == (4.0, 5.0, 6.0)


def test_distance_to():
    a = Atom(element="H", x=0.0, y=0.0, z=0.0)
    b = Atom(element="H", x=3.0, y=4.0, z=0.0)
    assert a.distance_to(b) == pytest.approx(5.0)


def test_distance_to_is_symmetric():
    a = Atom(element="H", x=1.0, y=2.0, z=3.0)
    b = Atom(element="O", x=-1.0, y=0.0, z=2.0)
    assert a.distance_to(b) == pytest.approx(b.distance_to(a))


def test_distance_to_self_is_zero():
    a = Atom(element="H", x=1.0, y=2.0, z=3.0)
    assert a.distance_to(a) == pytest.approx(0.0)
