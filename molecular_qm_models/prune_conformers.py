from typing import List, Union

import numpy as np

from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_models.align_molecules import align_molecules
from molecular_qm_models.internal_coordinates import InternalDihedralCoordinate, InternalCoordinatesList
from molecular_qm_models.molecular_geometry import Dihedral


def _molecule_energy(molecule: Molecule) -> float:
    """Return the energy of a molecule, requiring properties['energy'] to be set."""
    if "energy" not in molecule.properties:
        raise ValueError(
            "Each molecule must define properties['energy'] to be pruned."
        )
    return float(molecule.properties["energy"])


def prune_conformers(molecules: MoleculeList, rms_thresh: float) -> MoleculeList:
    """
    Prune conformers based on Cartesian RMSD.

    Molecules are ranked by ``properties["energy"]`` (lowest first) and any near
    duplicate that lies within ``rms_thresh`` (Angstrom) of an already-kept,
    lower-energy conformer is discarded. Before each pairwise comparison the two
    molecules are superimposed with the Kabsch algorithm (see
    :func:`align_molecules.align_molecules`) so the RMSD is translation- and
    rotation-invariant. Surviving molecules receive a ``properties["rank_id"]``
    that reflects their energy ordering.

    :param molecules: The conformers to prune. Each molecule must expose
        ``properties["energy"]`` and is assumed to share the same atom ordering.
        The list is sorted in place by energy as part of pruning.
    :param rms_thresh: RMSD threshold in Angstrom. Values <= 0 disable pruning.
    :return: A ``MoleculeList`` with the unique, energy-ranked conformers.
    """
    kept = MoleculeList()

    if len(molecules) == 0 or rms_thresh <= 0:
        for molecule in molecules:
            kept.append(molecule)
        return kept

    # Sort by energy to keep the lowest energy conformers. This also checks that
    # every molecule exposes properties["energy"].
    molecules.sort(key=_molecule_energy)

    for molecule in molecules:
        is_unique = True
        for existing in kept:
            # Align the candidate onto an already-kept conformer and compare the
            # resulting (best-fit) RMSD against the threshold.
            _, _, rmsd = align_molecules(existing, molecule)
            if rmsd < rms_thresh:
                is_unique = False
                break
        if is_unique:
            kept.append(molecule)

    # Assign a rank_id based on the (energy-sorted) surviving order.
    for rank_id, molecule in enumerate(kept):
        molecule.properties["rank_id"] = rank_id

    return kept


def _dihedral_values(
    molecule: Molecule, dihedrals: Union[InternalCoordinatesList, List[InternalDihedralCoordinate]]
) -> np.ndarray:
    """Return the dihedral angles (degrees) of ``molecule`` for each coordinate."""
    return np.array(
        [
            Dihedral.from_molecule(molecule, *dihedral.atom_indices)
            for dihedral in dihedrals
        ],
        dtype=float,
    )


def prune_conformers_by_angle(
    molecules: MoleculeList,
    dihedrals: Union[InternalCoordinatesList, List[InternalDihedralCoordinate]],
    angle_thresh: float = 15.0,
) -> MoleculeList:
    """
    Prune conformers based on their dihedral (torsion) angles.

    For every molecule the value of each supplied
    :class:`~molecular_qm_models.internal_coordinates.InternalDihedralCoordinate`
    is evaluated (in degrees). Molecules are ranked by ``properties["energy"]``
    (lowest first) and any conformer whose dihedral fingerprint lies within
    ``angle_thresh`` of an already-kept, lower-energy conformer is discarded. The
    similarity metric is the root-mean-square of the per-dihedral differences,
    computed with proper angular wrapping so that e.g. ``-179`` and ``179`` are
    treated as ~2 degrees apart. Surviving molecules receive a
    ``properties["rank_id"]`` reflecting their energy ordering.

    :param molecules: The conformers to prune. Each molecule must expose
        ``properties["energy"]`` and share the same atom ordering. The list is
        sorted in place by energy as part of pruning.
    :param dihedrals: The dihedral coordinates that define the torsional
        fingerprint used to compare conformers.
    :param angle_thresh: RMS dihedral difference threshold in degrees. Values
        <= 0 disable pruning.
    :return: A ``MoleculeList`` with the unique, energy-ranked conformers.
    """
    kept = MoleculeList()

    if len(molecules) == 0 or angle_thresh <= 0 or not dihedrals:
        for molecule in molecules:
            kept.append(molecule)
        return kept

    # Sort by energy to keep the lowest energy conformers. This also checks that
    # every molecule exposes properties["energy"].
    molecules.sort(key=_molecule_energy)

    kept_angles: List[np.ndarray] = []
    for molecule in molecules:
        angles = _dihedral_values(molecule, dihedrals)
        is_unique = True
        for existing_angles in kept_angles:
            # Wrap the per-dihedral differences into [0, 180] before taking the
            # RMS so that the metric is periodic in 360 degrees.
            diffs = np.abs(angles - existing_angles) % 360.0
            diffs = np.where(diffs > 180.0, 360.0 - diffs, diffs)
            rms = float(np.sqrt(np.mean(np.square(diffs))))
            if rms < angle_thresh:
                is_unique = False
                break
        if is_unique:
            kept.append(molecule)
            kept_angles.append(angles)

    # Assign a rank_id based on the (energy-sorted) surviving order.
    for rank_id, molecule in enumerate(kept):
        molecule.properties["rank_id"] = rank_id

    return kept
