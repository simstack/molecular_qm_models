import numpy as np
from typing import Tuple

from molecular_qm_models import Molecule


def _coords(molecule: Molecule) -> np.ndarray:
    """Return the (N, 3) array of atomic coordinates for a molecule."""
    return np.array([[atom.x, atom.y, atom.z] for atom in molecule.atoms], dtype=float)


def _with_coords(molecule: Molecule, coords: np.ndarray) -> Molecule:
    """Return a copy of ``molecule`` whose atom positions are ``coords``."""
    aligned = Molecule.from_molecule(molecule)
    for atom, xyz in zip(aligned.atoms, coords):
        atom.position = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return aligned


def align_molecules(reference: Molecule, mobile: Molecule) -> Tuple[Molecule, Molecule, float]:
    """
    Align two molecules using the Kabsch algorithm.

    The reference molecule is translated so that its centroid lies at the
    origin, and the mobile molecule is translated and rotated so that it
    optimally overlaps the reference (minimising the RMSD). Both molecules are
    assumed to share the same atom count and ordering.

    The returned molecules are independent copies; the inputs are left
    unchanged.

    :param reference: The reference molecule.
    :param mobile: The molecule to be rotated/translated onto the reference.
    :return: A tuple ``(aligned_reference, aligned_mobile, rmsd)`` where the two
        molecules are centred copies superimposed on each other and ``rmsd`` is
        the minimal root-mean-square deviation (Angstrom) between them.
    """
    ref_coords = _coords(reference)
    mobile_coords = _coords(mobile)

    if ref_coords.shape != mobile_coords.shape:
        raise ValueError(
            f"Cannot align molecules with different shapes: "
            f"{ref_coords.shape} vs {mobile_coords.shape}"
        )

    # Center both coordinate sets on their centroids.
    ref_centered = ref_coords - ref_coords.mean(axis=0)
    mobile_centered = mobile_coords - mobile_coords.mean(axis=0)

    # Optimal rotation of the mobile set onto the reference via SVD of the
    # covariance matrix.
    covariance = mobile_centered.T @ ref_centered
    v, _, wt = np.linalg.svd(covariance)
    # Correct for a possible reflection so we get a proper rotation.
    d = np.sign(np.linalg.det(v @ wt))
    correction = np.diag([1.0, 1.0, d])
    rotation = v @ correction @ wt

    mobile_aligned = mobile_centered @ rotation

    diff = mobile_aligned - ref_centered
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    return (
        _with_coords(reference, ref_centered),
        _with_coords(mobile, mobile_aligned),
        rmsd,
    )
