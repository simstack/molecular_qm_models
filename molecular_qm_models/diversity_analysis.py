from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner

from simstack.models.charts_artifact import (
    ChartArtifactModel,
    create_simple_bar_chart,
    create_simple_scatter_chart,
)

from .align_molecules import align_molecules
from .internal_coordinates import InternalDihedralCoordinate, InternalCoordinatesList
from .molecular_geometry import Dihedral
from .molecule import Molecule, MoleculeList


def _molecule_energy(molecule: Molecule) -> Optional[float]:
    """Return ``properties["energy"]`` of a molecule, or ``None`` if unset."""
    if "energy" not in molecule.properties:
        return None
    return float(molecule.properties["energy"])


def _energies(molecules: MoleculeList) -> Optional[List[float]]:
    """Return the energies of ``molecules`` if every molecule defines one."""
    energies = [_molecule_energy(molecule) for molecule in molecules]
    if any(energy is None for energy in energies):
        return None
    return [float(energy) for energy in energies]


def _histogram_rows(values: List[float], bins: int = 20) -> List[dict]:
    """Bin ``values`` and return AG-Charts rows with a bin label and a count."""
    if not values:
        return []
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    rows = []
    for i, count in enumerate(counts):
        center = 0.5 * (edges[i] + edges[i + 1])
        rows.append({"bin": f"{center:.2f}", "count": int(count)})
    return rows


def _pca_2d(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project ``features`` (n_samples, n_features) onto its first components.

    A plain numpy SVD is used (no scikit-learn dependency). Returns the scores
    for up to two principal components together with the explained variance
    ratio of each returned component.
    """
    centered = features - features.mean(axis=0)
    # full_matrices=False keeps the economy-size decomposition.
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u * s
    n_components = min(2, scores.shape[1])
    scores = scores[:, :n_components]
    total = float(np.sum(s ** 2))
    if total > 0.0:
        explained = (s[:n_components] ** 2) / total
    else:
        explained = np.zeros(n_components, dtype=float)
    return scores, explained


def _pca_scatter_chart(
    scores: np.ndarray, explained: np.ndarray, title: str
) -> ChartArtifactModel:
    """Build a PC1/PC2 scatter chart artifact from PCA ``scores``."""
    rows = []
    for index, score in enumerate(scores):
        row = {"conformer_index": index, "pc1": float(score[0])}
        row["pc2"] = float(score[1]) if score.shape[0] > 1 else 0.0
        rows.append(row)
    return create_simple_scatter_chart(rows, "pc1", "pc2", title)


@node
def analyze_conformer_diversity(molecules: MoleculeList, **kwargs) -> NodeRunner:
    """
    Characterise the diversity of a population of conformers in Cartesian space.

    Each molecule is treated as one conformer that shares the same atom ordering
    (see :func:`prune_conformers.prune_conformers`). The analysis superimposes
    every conformer onto the first one with the Kabsch algorithm
    (:func:`align_molecules.align_molecules`) and attaches, as
    :class:`~simstack.models.charts_artifact.ChartArtifactModel` figures on the
    ``node_runner``:

    * ``chart_pca_cartesian`` – a PCA scatter of the aligned Cartesian
      coordinates (PC1 vs PC2),
    * ``chart_pairwise_rmsd`` – a histogram of the pairwise best-fit RMSD
      (Angstrom), and
    * ``chart_energy`` – a histogram of the conformer energies (only when every
      molecule exposes ``properties["energy"]``).

    :param molecules: The conformers to analyse. They are assumed to share the
        same atom count and ordering.
    :return: The finalised :class:`~simstack.core.node_runner.NodeRunner` with
        the chart artifacts attached. No charts are attached when there are
        fewer than two conformers.
    """
    node_runner: NodeRunner = kwargs["node_runner"]
    node_runner.log(f"Analyzing conformer diversity for {len(molecules)} conformers")

    n = len(molecules)
    if n < 2:
        return node_runner.succeed()

    reference = molecules[0]

    # Align every conformer onto the first and collect the centred coordinates
    # plus the best-fit RMSD against the reference.
    aligned_coords: List[np.ndarray] = []
    for molecule in molecules:
        _, aligned_mobile, _ = align_molecules(reference, molecule)
        coords = np.array(
            [[atom.x, atom.y, atom.z] for atom in aligned_mobile.atoms], dtype=float
        )
        aligned_coords.append(coords.flatten())

    features = np.array(aligned_coords)
    scores, explained = _pca_2d(features)
    node_runner.chart_pca_cartesian = _pca_scatter_chart(
        scores, explained, "Conformer Diversity PCA (Cartesian)"
    )

    # Pairwise best-fit RMSD histogram.
    rmsd_vals: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            _, _, rmsd = align_molecules(molecules[i], molecules[j])
            rmsd_vals.append(float(rmsd))
    node_runner.chart_pairwise_rmsd = create_simple_bar_chart(
        _histogram_rows(rmsd_vals),
        "bin",
        "count",
        "Pairwise RMSD Distribution (Angstrom)",
    )

    # Energy histogram (only if every conformer exposes an energy).
    energies = _energies(molecules)
    if energies is not None:
        node_runner.chart_energy = create_simple_bar_chart(
            _histogram_rows(energies),
            "bin",
            "count",
            "Energy Distribution (kcal/mol)",
        )

    return node_runner.succeed()


@node
def analyze_conformer_diversity_by_dihedrals(
    molecules: MoleculeList,
    dihedrals: InternalCoordinatesList,
    **kwargs,
) -> NodeRunner:
    """
    Characterise conformer diversity using their dihedral (torsion) angles.

    For every molecule the value of each supplied
    :class:`~molecular_qm_models.internal_coordinates.InternalDihedralCoordinate`
    is evaluated (in degrees). The analysis attaches, as
    :class:`~simstack.models.charts_artifact.ChartArtifactModel` figures on the
    ``node_runner``:

    * ``chart_pca_dihedral`` – a PCA scatter of the dihedral fingerprints
      (PC1 vs PC2), computed from the ``sin``/``cos`` of every angle so that the
      metric is periodic,
    * ``chart_pairwise_dihedral_rmsd`` – a histogram of the pairwise dihedral
      RMSD (degrees) using proper angular wrapping, and
    * ``chart_energy`` – a histogram of the conformer energies (only when every
      molecule exposes ``properties["energy"]``).

    :param molecules: The conformers to analyse. They are assumed to share the
        same atom ordering.
    :param dihedrals: The dihedral coordinates that define the torsional
        fingerprint used to compare conformers.
    :return: The finalised :class:`~simstack.core.node_runner.NodeRunner` with
        the chart artifacts attached. No charts are attached when there are
        fewer than two conformers or no dihedrals are supplied.
    """
    node_runner: NodeRunner = kwargs["node_runner"]
    node_runner.log(
        f"Analyzing dihedral conformer diversity for {len(molecules)} conformers"
    )

    n = len(molecules)
    if n < 2 or not dihedrals:
        return node_runner.succeed()

    # Evaluate every dihedral (degrees) for every conformer -> (n, n_dihedrals).
    angles = np.array(
        [
            [
                Dihedral.from_molecule(molecule, *dihedral.atom_indices)
                for dihedral in dihedrals
            ]
            for molecule in molecules
        ],
        dtype=float,
    )

    # PCA on sin/cos features to respect the periodicity of the angles.
    rad = np.deg2rad(angles)
    features = np.column_stack([np.sin(rad), np.cos(rad)])
    scores, explained = _pca_2d(features)
    node_runner.chart_pca_dihedral = _pca_scatter_chart(
        scores, explained, "Conformer Diversity PCA (Dihedral)"
    )

    # Vectorised all-to-all dihedral RMSD with angular wrapping into [0, 180].
    diffs = np.abs(angles[:, np.newaxis, :] - angles[np.newaxis, :, :]) % 360.0
    diffs = np.where(diffs > 180.0, 360.0 - diffs, diffs)
    rmsd_matrix = np.sqrt(np.mean(np.square(diffs), axis=-1))
    rmsd_vals = rmsd_matrix[np.triu_indices(n, k=1)].tolist()
    node_runner.chart_pairwise_dihedral_rmsd = create_simple_bar_chart(
        _histogram_rows(rmsd_vals),
        "bin",
        "count",
        "Pairwise Dihedral RMSD Distribution (degrees)",
    )

    # Energy histogram (only if every conformer exposes an energy).
    energies = _energies(molecules)
    if energies is not None:
        node_runner.chart_energy = create_simple_bar_chart(
            _histogram_rows(energies),
            "bin",
            "count",
            "Energy Distribution (kcal/mol)",
        )

    return node_runner.succeed()
