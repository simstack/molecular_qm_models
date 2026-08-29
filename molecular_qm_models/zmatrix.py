import numpy as np
from typing import List, Optional
from molecular_qm_models import Molecule, Atom

class ZMatrix:
    """
    A class representing a molecule in internal coordinates (Z-matrix).
    """
    def __init__(self):
        self.atoms = []  # List of (element, [ref_atoms], [values])
        # values are (distance, angle, dihedral)

    def add_atom(self, element: str, 
                 ref_atom1: Optional[int] = None, distance: Optional[float] = None,
                 ref_atom2: Optional[int] = None, angle: Optional[float] = None,
                 ref_atom3: Optional[int] = None, dihedral: Optional[float] = None):
        """
        Add an atom to the Z-matrix.
        Angles and dihedrals are in degrees.
        """
        self.atoms.append({
            'element': element,
            'refs': [ref_atom1, ref_atom2, ref_atom3],
            'values': [distance, angle, dihedral]
        })

    def to_cartesian(self) -> Molecule:
        """
        Convert Z-matrix to Cartesian coordinates (Molecule object).
        """
        cart_coords = []
        molecule = Molecule()

        for i, atom_data in enumerate(self.atoms):
            element = atom_data['element']
            refs = atom_data['refs']
            vals = atom_data['values']

            if i == 0:
                pos = np.array([0.0, 0.0, 0.0])
            elif i == 1:
                pos = np.array([vals[0], 0.0, 0.0])
            elif i == 2:
                # Atom 2 is in XY plane
                r = vals[0]
                theta = np.radians(vals[1])
                ref0 = cart_coords[refs[0]]
                ref1 = cart_coords[refs[1]]
                
                # Simple placement for the 3rd atom
                pos = ref0 + np.array([r * np.cos(np.pi - theta), r * np.sin(np.pi - theta), 0.0])
            else:
                r = vals[0]
                theta = np.radians(vals[1])
                phi = np.radians(vals[2])
                
                i_ref = refs[0]
                j_ref = refs[1]
                k_ref = refs[2]
                
                p_i = cart_coords[i_ref]
                p_j = cart_coords[j_ref]
                p_k = cart_coords[k_ref]
                
                pos = self._calculate_position(p_i, p_j, p_k, r, theta, phi)
            
            cart_coords.append(pos)
            molecule.add_atom(Atom(element=element, x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
            
        return molecule

    def _calculate_position(self, p_i, p_j, p_k, r, theta, phi):
        """
        Calculate position of atom from 3 reference atoms, distance, angle and dihedral.
        p_i: position of atom it's bonded to
        p_j: position of atom forming angle with p_i
        p_k: position of atom forming dihedral with p_i, p_j
        """
        v1 = p_i - p_j
        v2 = p_k - p_j
        
        # Normal to the plane i-j-k
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-10:
            # Collinear refs, try another vector
            v2 = v1 + np.array([1.0, 0.0, 0.0])
            n = np.cross(v1, v2)
            if np.linalg.norm(n) < 1e-10:
                v2 = v1 + np.array([0.0, 1.0, 0.0])
                n = np.cross(v1, v2)

        n /= np.linalg.norm(n)
        
        # Plane vector
        v1_u = v1 / np.linalg.norm(v1)
        m = np.cross(n, v1_u)
        
        # Position relative to p_i
        # theta is angle j-i-new, but Z-matrix standard is usually i-j-new?
        # Actually standard Z-matrix: 
        # new is at distance r from i
        # angle(new-i-j) is theta
        # dihedral(new-i-j-k) is phi
        
        # Rotation for angle
        # Start along -v1 (away from j)
        direction = -v1_u * np.cos(theta) + m * np.sin(theta)
        
        # Rotation for dihedral (around v1_u)
        # We need a vector perpendicular to v1_u to rotate
        # m is already perpendicular to v1_u and in the plane i-j-k
        # n is perpendicular to the plane
        
        # The vector m is at dihedral 0 or 180?
        # Let's use the Neuman projection-like construction
        
        # Standard way:
        # x-axis along i->j
        # y-axis in i-j-k plane
        # z-axis normal
        
        a = -v1_u
        b = np.cross(v1, v2)
        b /= np.linalg.norm(b)
        c = np.cross(b, a)
        
        # pos = p_i + r * (a * cos(theta) + b * sin(theta) * sin(phi) + c * sin(theta) * cos(phi))
        # Wait, standard Z-matrix dihedral definition:
        # phi is angle between plane (new, i, j) and (i, j, k)
        
        pos = p_i + r * (a * np.cos(theta) + 
                         c * np.sin(theta) * np.cos(phi) + 
                         b * np.sin(theta) * np.sin(phi))
        
        return pos

    def set_value(self, atom_index: int, value_index: int, value: float):
        """
        Set a specific value (0=dist, 1=angle, 2=dihedral) for an atom.
        """
        self.atoms[atom_index]['values'][value_index] = value

    def get_value(self, atom_index: int, value_index: int) -> float:
        return self.atoms[atom_index]['values'][value_index]

    @classmethod
    def from_trajectories(cls, trajectories: List[List[Molecule]]) -> 'ZMatrix':
        """
        Constructs an optimal Z-matrix representation by analyzing multiple NEB trajectories.
        Identifies a stable atom numbering and reference connectivity based on distance variances.
        """
        all_molecules = [mol for traj in trajectories for mol in traj]
        if not all_molecules:
            return cls()

        num_atoms = len(all_molecules[0].atoms)
        
        # 1. Compute average distance matrix
        avg_dist_matrix = np.zeros((num_atoms, num_atoms))
        for mol in all_molecules:
            coords = np.array([atom.position for atom in mol.atoms])
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            avg_dist_matrix += dist
        avg_dist_matrix /= len(all_molecules)

        # 2. Compute distance variances (stability of bonds)
        dist_variances = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dists = [np.linalg.norm(np.array(mol.atoms[i].position) - 
                                        np.array(mol.atoms[j].position)) for mol in all_molecules]
                dist_variances[i, j] = dist_variances[j, i] = np.var(dists)

        # 3. Choose starting atom (most stable environment)
        stability_scores = []
        for i in range(num_atoms):
            neighbors = np.argsort(avg_dist_matrix[i])[1:4]
            score = np.mean(dist_variances[i, neighbors])
            stability_scores.append(score)
        
        start_atom = np.argmin(stability_scores)
        
        zmatrix = cls()
        ordered_indices = [start_atom]
        remaining_indices = [i for i in range(num_atoms) if i != start_atom]
        
        zmatrix.add_atom(all_molecules[0].atoms[start_atom].element)

        # 4. Iteratively add atoms
        while remaining_indices:
            best_next_val = float('inf')
            best_next_idx = -1
            best_ref1 = -1
            
            for rem_idx in remaining_indices:
                for ord_idx in ordered_indices:
                    score = dist_variances[rem_idx, ord_idx]
                    if score < best_next_val:
                        best_next_val = score
                        best_next_idx = rem_idx
                        best_ref1 = ord_idx
            
            if best_next_idx == -1 or best_next_val == 0:
                best_next_val = float('inf')
                for rem_idx in remaining_indices:
                    for ord_idx in ordered_indices:
                        d = avg_dist_matrix[rem_idx, ord_idx]
                        if d < best_next_val:
                            best_next_val = d
                            best_next_idx = rem_idx
                            best_ref1 = ord_idx
            
            ref_idx1 = ordered_indices.index(best_ref1)
            ref_idx2 = None
            if len(ordered_indices) >= 2:
                other_ordered = [idx for idx in ordered_indices if idx != best_ref1]
                best_ref2 = other_ordered[np.argmin(avg_dist_matrix[best_ref1, other_ordered])]
                ref_idx2 = ordered_indices.index(best_ref2)
                
            ref_idx3 = None
            if len(ordered_indices) >= 3:
                other_ordered = [idx for idx in ordered_indices if idx not in [best_ref1, best_ref2]]
                best_ref3 = other_ordered[np.argmin(avg_dist_matrix[best_ref2, other_ordered])]
                ref_idx3 = ordered_indices.index(best_ref3)
                
            # Average values
            avg_r, avg_theta, avg_phi = 0, 0, 0
            for mol in all_molecules:
                p_curr = np.array(mol.atoms[best_next_idx].position)
                p_ref1 = np.array(mol.atoms[best_ref1].position)
                avg_r += np.linalg.norm(p_curr - p_ref1)
                
                if ref_idx2 is not None:
                    p_ref2 = np.array(mol.atoms[ordered_indices[ref_idx2]].position)
                    v1, v2 = p_curr - p_ref1, p_ref2 - p_ref1
                    avg_theta += np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1)))
                    
                    if ref_idx3 is not None:
                        p_ref3 = np.array(mol.atoms[ordered_indices[ref_idx3]].position)
                        avg_phi += cls._calculate_dihedral(p_curr, p_ref1, p_ref2, p_ref3)
            
            zmatrix.add_atom(all_molecules[0].atoms[best_next_idx].element,
                             ref_atom1=ref_idx1, distance=avg_r/len(all_molecules),
                             ref_atom2=ref_idx2, angle=avg_theta/len(all_molecules),
                             ref_atom3=ref_idx3, dihedral=avg_phi/len(all_molecules))
            
            ordered_indices.append(best_next_idx)
            remaining_indices.remove(best_next_idx)
            
        return zmatrix

    @staticmethod
    def _calculate_dihedral(p1, p2, p3, p4):
        b0, b1, b2 = -1.0*(p2 - p1), p3 - p2, p4 - p3
        b1 /= np.linalg.norm(b1)
        v, w = b0 - np.dot(b0, b1)*b1, b2 - np.dot(b2, b1)*b1
        return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
