"""Tests for :class:`molecular_qm_models.QMInput` fields used by nested QM nodes."""

from molecular_qm_models import BasisSet, Functional, QMInput


def test_name_defaults_to_none(water):
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
    )
    assert qm_input.name is None


def test_name_can_be_assigned_after_construction(water):
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
    )
    qm_input.name = "ground_state.1-(3-hydroxyphenyl)ethanone"
    assert qm_input.name == "ground_state.1-(3-hydroxyphenyl)ethanone"


def test_name_survives_dump_and_reconstruct(water):
    original = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
        name="excited_state_scan.water",
    )
    dumped = original.model_dump(exclude={"id"})
    restored = QMInput(**dumped)
    assert restored.name == "excited_state_scan.water"


def test_legacy_dump_without_name_stays_unset(water):
    dumped = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
    ).model_dump(exclude={"id"})
    dumped.pop("name", None)
    restored = QMInput(**dumped)
    assert restored.name is None


def test_json_schema_exposes_name_as_string():
    schema = QMInput.json_schema()
    name_schema = schema["properties"]["name"]
    assert name_schema["type"] == "string"
    assert "anyOf" not in name_schema
    assert "name" in QMInput.ui_schema()["ui:order"]
