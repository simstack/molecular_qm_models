import numpy as np
from molecular_qm_util.rdkit_scripts import get_rotatable_bonds
from typing import List

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

from .models import RankedConformer


try:
    from openbabel import openbabel as ob
except ImportError:
    ob = None


def prune_conformers(ranked: List[RankedConformer], rms_thresh: float) -> List[RankedConformer]:
    """
    Prune conformers based on RMSD.
    Uses Dihedral RMSD if rotatable bonds are available for speed.
    """
    if not ranked or rms_thresh <= 0:
        return ranked

    # Sort by energy to keep the lowest energy conformers
    ranked_sorted = sorted(ranked, key=lambda x: x.energy_kcal_mol)
    n = len(ranked_sorted)

    # Check if we can use dihedral pruning
    # Assume all molecules in the list are the same structure
    first_mol = ranked_sorted[0].mol
    if not isinstance(first_mol, Molecule):
        # Fallback to existing logic for non-RDKit molecules (e.g. OpenBabel)
        return _prune_conformers_cartesian(ranked_sorted, rms_thresh)

    dihedrals = get_rotatable_bonds(first_mol)
    if not dihedrals:
        # Fallback to Cartesian RMSD if no rotatable bonds
        return _prune_conformers_cartesian(ranked_sorted, rms_thresh)

    # Extract dihedral angles for all conformers
    angles = []
    for rc in ranked_sorted:
        conf = rc.mol.GetConformer(rc.conf_id)
        row = [AllChem.GetDihedralDeg(conf, *d) for d in dihedrals]
        angles.append(row)
    
    data = np.array(angles)
    
    pruned_indices = []
    
    # Use 15.0 degrees as a base for 0.1 A? Or just use the thresh * 100?
    # Actually, let's just use a reasonable default if it looks like Angstroms.
    effective_thresh = rms_thresh
    if rms_thresh < 2.0: # Likely Angstroms
        effective_thresh = rms_thresh * 150.0 # 0.1 A -> 15 degrees
    
    for i in range(n):
        is_unique = True
        for j in pruned_indices:
            # Dihedral RMSD calculation
            diffs = np.abs(data[i] - data[j]) % 360
            diffs = np.where(diffs > 180, 360 - diffs, diffs)
            rms = np.sqrt(np.mean(np.square(diffs)))
            
            if rms < effective_thresh:
                is_unique = False
                break
        if is_unique:
            pruned_indices.append(i)
            
    return [ranked_sorted[i] for i in pruned_indices]


def _prune_conformers_cartesian(ranked_sorted: List[RankedConformer], rms_thresh: float) -> List[RankedConformer]:
    pruned: List[RankedConformer] = []
    for rc in ranked_sorted:
        is_unique = True
        for existing in pruned:
            if isinstance(rc.mol, Chem.Mol) and isinstance(existing.mol, Chem.Mol):
                rms = AllChem.GetBestRMS(rc.mol, existing.mol, rc.conf_id, existing.conf_id, maxMatches=1000)
            elif hasattr(rc.mol, "OBMol") and hasattr(existing.mol, "OBMol"):
                align = ob.OBAlign(False, True)
                align.SetRefMol(existing.mol.OBMol)
                align.SetTargetMol(rc.mol.OBMol)
                align.Align()
                rms = align.GetRMSD()
            else:
                raise RuntimeError(f"Unsupported molecule type: {type(rc.mol)} vs {type(existing.mol)}")

            if rms < rms_thresh:
                is_unique = False
                break
        if is_unique:
            pruned.append(rc)
    return pruned
