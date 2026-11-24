"""Tests for AnnotatedMolecule sphere and bond annotations."""

import pytest

from molecular_qm_models import AnnotatedMolecule, BondAnnotation, Molecule, SphereAnnotation


def test_annotated_molecule_stores_sphere_and_bond_annotations(water):
    annotated = AnnotatedMolecule(
        molecule=water,
        sphere_annotations=[
            SphereAnnotation(coordinate=[1.0, 0.0, 0.0], radius=0.4, color="#ff0000", transparency=0.25)
        ],
        bond_annotations=[
            BondAnnotation(
                points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                color="blue",
                thickness=0.15,
                transparency=0.0,
            )
        ],
    )

    assert annotated.molecule is water
    assert annotated.sphere_annotations[0].coordinate == [1.0, 0.0, 0.0]
    assert annotated.bond_annotations[0].points[1] == [1.0, 0.0, 0.0]
    assert annotated.ui_schema()["ui:field"] == "AnnotatedMoleculeField"


def test_annotation_lists_default_to_empty(water):
    annotated = AnnotatedMolecule(molecule=water)
    assert annotated.sphere_annotations == []
    assert annotated.bond_annotations == []


def test_molecule_reference_is_required():
    with pytest.raises(ValueError):
        AnnotatedMolecule()


def test_null_annotation_lists_are_rejected(water):
    with pytest.raises(ValueError):
        AnnotatedMolecule(molecule=water, sphere_annotations=None)
    with pytest.raises(ValueError):
        AnnotatedMolecule(molecule=water, bond_annotations=None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coordinate": [1.0, 2.0], "radius": 1.0, "color": "red", "transparency": 0.0},
        {"coordinate": [1.0, 2.0, float("nan")], "radius": 1.0, "color": "red", "transparency": 0.0},
        {"coordinate": [0.0, 0.0, 0.0], "radius": 0.0, "color": "red", "transparency": 0.0},
        {"coordinate": [0.0, 0.0, 0.0], "radius": 1.0, "color": "  ", "transparency": 0.0},
        {"coordinate": [0.0, 0.0, 0.0], "radius": 1.0, "color": "red", "transparency": 1.5},
    ],
)
def test_sphere_annotation_rejects_uncontrolled_values(kwargs):
    with pytest.raises(ValueError):
        SphereAnnotation(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": [[0.0, 0.0, 0.0]], "color": "red", "thickness": 0.1, "transparency": 0.0},
        {
            "points": [[0.0, 0.0, 0.0], [1.0, 0.0]],
            "color": "red",
            "thickness": 0.1,
            "transparency": 0.0,
        },
        {
            "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "color": "red",
            "thickness": -0.2,
            "transparency": 0.0,
        },
        {
            "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "color": "",
            "thickness": 0.2,
            "transparency": 0.5,
        },
    ],
)
def test_bond_annotation_rejects_uncontrolled_values(kwargs):
    with pytest.raises(ValueError):
        BondAnnotation(**kwargs)


def test_annotated_molecule_accepts_a_molecule_built_from_sites():
    molecule = Molecule.from_sites(["C"], [[0.0, 0.0, 0.0]])
    annotated = AnnotatedMolecule(
        molecule=molecule,
        sphere_annotations=[
            SphereAnnotation(coordinate=[0.0, 0.0, 1.0], radius=0.8, color="green", transparency=1.0)
        ],
    )
    assert len(annotated.molecule.atoms) == 1
    assert annotated.sphere_annotations[0].transparency == 1.0
