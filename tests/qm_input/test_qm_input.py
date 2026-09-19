"""Tests for QMInput excited-state field visibility and validation."""

from molecular_qm_models import BasisSet, Functional, QMInput, QMMethod


def _method_options(condition):
    method = condition["method"]
    if isinstance(method, dict):
        return list(method.get("ui:options") or method.get("enum") or [])
    if isinstance(method, list):
        return list(method)
    return []


def test_ui_schema_shows_focus_state_with_the_excited_states_checkbox():
    ui = QMInput.ui_schema()
    assert ui["states"]["ui:condition"] == {"excited_states": True}
    assert ui["focus_state"]["ui:condition"] == {"excited_states": True}


def test_ui_schema_forces_excited_states_on_for_tddft_and_casscf():
    ui = QMInput.ui_schema()
    options = _method_options(ui["excited_states"]["ui:disabledCondition"])
    assert "TDDFT" in options
    assert "CASSCF" in options
    assert ui["excited_states"]["ui:disabledValue"] is True


def test_ui_schema_shows_active_space_for_casscf():
    ui = QMInput.ui_schema()
    for field in ("active_orbitals", "active_electrons"):
        options = _method_options(ui[field]["ui:condition"])
        assert "CASSCF" in options
        assert "DFTMRCI" in options


def test_json_schema_method_branches_include_states_and_focus_state():
    tddft_branch = None
    casscf_branch = None
    for branch in QMInput.json_schema()["dependencies"]["method"]["oneOf"]:
        method_schema = branch.get("properties", {}).get("method", {})
        if method_schema.get("enum") == ["TDDFT"]:
            tddft_branch = branch
        elif method_schema.get("enum") == ["CASSCF", "DFTMRCI"]:
            casscf_branch = branch
    assert tddft_branch is not None
    assert casscf_branch is not None

    tddft_properties = tddft_branch["properties"]
    assert "states" in tddft_properties
    assert "focus_state" in tddft_properties
    assert "functional" in tddft_properties
    assert "active_orbitals" not in tddft_properties
    assert "active_electrons" not in tddft_properties

    casscf_properties = casscf_branch["properties"]
    assert "states" in casscf_properties
    assert "focus_state" in casscf_properties
    assert "active_orbitals" in casscf_properties
    assert "active_electrons" in casscf_properties


def test_tddft_keeps_states_when_checkbox_is_off(water):
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
        method=QMMethod.TDDFT,
        states=10,
        focus_state=2,
        excited_states=False,
    )
    assert qm_input.states == 10
    assert qm_input.focus_state == 2


def test_casscf_keeps_states_when_checkbox_is_off(water):
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
        method=QMMethod.CASSCF,
        states=8,
        focus_state=3,
        active_electrons=6,
        active_orbitals=6,
        excited_states=False,
    )
    assert qm_input.states == 8
    assert qm_input.focus_state == 3
    assert qm_input.active_electrons == 6
    assert qm_input.active_orbitals == 6


def test_dft_without_excited_states_clears_states(water):
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="def2-SVP"),
        functional=Functional(functional="B3LYP"),
        method=QMMethod.DFT,
        states=10,
        focus_state=2,
        excited_states=False,
    )
    assert qm_input.states == 0
    assert qm_input.focus_state == 1
