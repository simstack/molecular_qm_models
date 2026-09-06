"""Tests for :mod:`molecular_qm_models.diversity_analysis`.

Both public functions are simstack ``@node`` functions: they attach their
figures as :class:`~simstack.models.charts_artifact.ChartArtifactModel`
artifacts onto the injected ``node_runner`` (never matplotlib figures) and
finish by returning ``node_runner.succeed()``. For degenerate inputs no chart
artifacts are attached.

The node body is exercised directly through the wrapper's ``_inner`` attribute
so no database/execution context is required.
"""

from typing import Dict, List

import pytest

from molecular_qm_models import (
    InternalCoordinatesList,
    InternalDihedralCoordinate,
    Molecule,
    MoleculeList,
    analyze_conformer_diversity,
    analyze_conformer_diversity_by_dihedrals,
)
from simstack.core.definitions import TaskStatus
from simstack.core.node_runner import NodeRunner
from simstack.models.charts_artifact import ChartArtifactModel


CHAIN_ELEMENTS: List[str] = ["C", "C", "O", "H"]
CHAIN_SITES: List[List[float]] = [
    [0.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [2.0, 1.2, 0.0],
    [3.0, 1.2, 0.5],
]


def _make_conformers(count: int, with_energy: bool = True) -> MoleculeList:
    """Build ``count`` slightly perturbed C-C-O-H conformers."""
    molecules = MoleculeList()
    for k in range(count):
        # Rotate the terminal atom around the 1-2-3 axis so the dihedral changes.
        sites = [list(site) for site in CHAIN_SITES]
        sites[3][2] = 0.5 + 0.4 * k
        molecule = Molecule.from_sites(list(CHAIN_ELEMENTS), sites)
        if with_energy:
            molecule.properties["energy"] = float(k)
        molecules.append(molecule)
    return molecules


def _runner() -> NodeRunner:
    """Create a bare NodeRunner suitable for driving a node body in tests."""
    return NodeRunner("test_diversity", "task-diversity")


def _chart_artifacts(node_runner: NodeRunner) -> Dict[str, ChartArtifactModel]:
    """Return the ``chart_*`` attributes attached to ``node_runner``."""
    extra = node_runner.model_extra or {}
    return {
        name: value
        for name, value in extra.items()
        if name.startswith("chart_") and isinstance(value, ChartArtifactModel)
    }


def test_analyze_conformer_diversity_returns_chart_artifacts():
    node_runner = _runner()

    result = analyze_conformer_diversity._inner(
        _make_conformers(5), node_runner=node_runner
    )

    assert result is node_runner
    assert node_runner.status == TaskStatus.COMPLETED

    charts = _chart_artifacts(node_runner)
    assert charts, "expected at least one chart artifact"
    assert "chart_pca_cartesian" in charts
    assert "chart_pairwise_rmsd" in charts
    assert "chart_energy" in charts

    titles = [chart.title.text for chart in charts.values()]
    assert "Conformer Diversity PCA (Cartesian)" in titles
    assert any("RMSD" in title for title in titles)
    assert any("Energy" in title for title in titles)


def test_analyze_conformer_diversity_without_energy_skips_energy_chart():
    node_runner = _runner()

    analyze_conformer_diversity._inner(
        _make_conformers(4, with_energy=False), node_runner=node_runner
    )

    charts = _chart_artifacts(node_runner)
    assert "chart_energy" not in charts


def test_analyze_conformer_diversity_by_dihedrals_returns_chart_artifacts():
    node_runner = _runner()
    dihedral = InternalDihedralCoordinate.initialize(0, 1, 2, 3, 0.0, 360.0)
    ic_list = InternalCoordinatesList(elements=[dihedral])

    result = analyze_conformer_diversity_by_dihedrals._inner(
        _make_conformers(5), ic_list, node_runner=node_runner
    )

    assert result is node_runner
    assert node_runner.status == TaskStatus.COMPLETED

    charts = _chart_artifacts(node_runner)
    assert charts, "expected at least one chart artifact"
    assert "chart_pca_dihedral" in charts
    assert "chart_pairwise_dihedral_rmsd" in charts

    titles = [chart.title.text for chart in charts.values()]
    assert "Conformer Diversity PCA (Dihedral)" in titles
    assert any("Dihedral RMSD" in title for title in titles)


def test_analyze_conformer_diversity_by_dihedrals_accepts_raw_list():
    node_runner = _runner()
    dihedral = InternalDihedralCoordinate.initialize(0, 1, 2, 3, 0.0, 360.0)

    result = analyze_conformer_diversity_by_dihedrals._inner(
        _make_conformers(5), [dihedral], node_runner=node_runner
    )

    assert result is node_runner
    assert node_runner.status == TaskStatus.COMPLETED
    charts = _chart_artifacts(node_runner)
    assert "chart_pca_dihedral" in charts


@pytest.mark.parametrize("count", [0, 1])
def test_analyze_functions_attach_no_charts_for_too_few_conformers(count):
    molecules = _make_conformers(count)
    dihedral = InternalDihedralCoordinate.initialize(0, 1, 2, 3, 0.0, 360.0)

    cartesian_runner = _runner()
    analyze_conformer_diversity._inner(molecules, node_runner=cartesian_runner)
    assert _chart_artifacts(cartesian_runner) == {}
    assert cartesian_runner.status == TaskStatus.COMPLETED

    dihedral_runner = _runner()
    analyze_conformer_diversity_by_dihedrals._inner(
        molecules, [dihedral], node_runner=dihedral_runner
    )
    assert _chart_artifacts(dihedral_runner) == {}
    assert dihedral_runner.status == TaskStatus.COMPLETED


def test_analyze_by_dihedrals_attaches_no_charts_without_dihedrals():
    node_runner = _runner()

    analyze_conformer_diversity_by_dihedrals._inner(
        _make_conformers(3), [], node_runner=node_runner
    )

    assert _chart_artifacts(node_runner) == {}
    assert node_runner.status == TaskStatus.COMPLETED
