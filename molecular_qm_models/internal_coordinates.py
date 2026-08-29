from enum import Enum
from typing import List, Dict, Union, Annotated, Literal
import numpy as np

from pydantic import Field as PydanticField
from odmantic import EmbeddedModel, Model, Field

from .molecule import Molecule
from simstack.models.base_lists import GenericListMixin
from molecular_qm_models.molecular_geometry import Bond, Angle, Dihedral


class InternalCoordinateType(str, Enum):
    BOND = "bond"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"
    FRAGMENT = "complex"
    INTERPOLATOR = "interpolator"
    FRAGMENT_ATOM = "fragment_atom"

class InternalCoordinateBondType(str,Enum):
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    AROMATIC = "aromatic"
    HYDROGEN = "hydrogen"
    UNKNOWN = "unknown"

class InternalCoordinateBase(EmbeddedModel):
    type: InternalCoordinateType
    atom_indices: List[int]
    min_values: List[float]    
    max_values: List[float]
    real_values: List[float] = []
    value: float = 0.0
    moving_atoms: List[int] = [] # Indices of atoms that move when this coordinate is changed

    @staticmethod
    def get_angle(p1, p2, p3):
        return Angle(p1, p2, p3).compute()

    @staticmethod
    def get_distance(p1, p2):
        return Bond(p1, p2).compute()

    @staticmethod
    def get_dihedral(p1, p2, p3, p4):
        return Dihedral(p1, p2, p3, p4).compute()

    def compute(self, molecule: Molecule):
        pass

    def set(self, molecule: Molecule, target_value: float):
        """
        Sets the internal coordinate to the target_value by moving the atoms in moving_atoms.
        target_value is normalized (0 to 1).
        """
        pass

    def get_actual_value(self, target_norm_value: float, index: int = 0) -> float:
        """Converts normalized value [0,1] to actual physical value."""
        return self.min_values[index] + target_norm_value * (self.max_values[index] - self.min_values[index])

    @staticmethod
    def get_moving_atoms(molecule: Molecule, bond: tuple[int, int], collision_set: List[int] = None) -> List[int]:
        """
        Finds all atoms on one side of a bond. 
        Returns indices of atoms reachable from bond[1] without going through bond[0]
        and not in collision_set.
        """
        adj = InternalCoordinateBase._get_adjacency(molecule)
        root = bond[1]
        barrier = bond[0]
        
        visited = {barrier, root}
        if collision_set:
            visited.update(collision_set)
        
        stack = [root]
        moving = [root]
        
        while stack:
            curr = stack.pop()
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
                    moving.append(neighbor)
        
        # print(f"Moving atoms for bond {bond}: {moving}")
        return moving

    @staticmethod
    def _get_adjacency(molecule: Molecule, threshold: float = 1.5) -> List[List[int]]:
        # Covalent radii in Angstroms for first 2 rows of periodic table
        covalent_radii = {
            'H': 0.31, 'He': 0.28,
            'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58
        }

        num_atoms = len(molecule.atoms)

        adj = [[] for _ in range(num_atoms)]
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = molecule.atoms[i].distance_to(molecule.atoms[j])

                # Get covalent radii for both atoms, default to threshold if not found
                symbol_i = molecule.atoms[i].element
                symbol_j = molecule.atoms[j].element
                radius_i = covalent_radii.get(symbol_i, threshold / 2)
                radius_j = covalent_radii.get(symbol_j, threshold / 2)

                # Bond exists if distance is less than sum of covalent radii + tolerance
                bond_threshold = (radius_i + radius_j) * 1.2  # 20% tolerance

                if dist < bond_threshold:
                    adj[i].append(j)
                    adj[j].append(i)
        return adj

    def _check_non_moving_atoms_unchanged(self, molecule: Molecule, original_positions: dict, tolerance: float = 1e-8):
        """
        Checks that atoms not in the moving set have the same coordinates before and after.

        Args:
            molecule: The molecule after modification
            original_positions: Dictionary mapping atom indices to their original positions (as numpy arrays)
            tolerance: Maximum allowed position change for non-moving atoms
        """
        num_atoms = len(molecule.atoms)
        moving_set = set(self.moving_atoms)

        for idx in range(num_atoms):
            if idx not in moving_set and idx in original_positions:
                current_pos = np.array(molecule.atoms[idx].position)
                original_pos = original_positions[idx]
                diff = np.linalg.norm(current_pos - original_pos)
                if diff > tolerance:
                    raise ValueError(
                        f"Atom {idx} is not in moving_atoms but moved by {diff:.6e} "
                        f"(tolerance: {tolerance:.6e})"
                    )


class InternalBondCoordinate(InternalCoordinateBase):
    type: Literal[InternalCoordinateType.BOND] = InternalCoordinateType.BOND
    bond: InternalCoordinateBondType = InternalCoordinateBondType.UNKNOWN
    @classmethod
    def initialize(cls, atom1: int, atom2: int, min_value: float, max_value: float, molecule: Molecule = None):
        moving_atoms = []
        if molecule:
            moving_atoms = cls.get_moving_atoms(molecule, (atom1, atom2))
        return cls(type=InternalCoordinateType.BOND, atom_indices=[atom1, atom2],
                   min_values=[min_value], max_values=[max_value],
                   value=0, moving_atoms=moving_atoms)

    def __str__(self):
        atom_str = '-'.join(str(idx + 1) for idx in self.atom_indices)
        actual_value = self.get_actual_value(self.value)
        return (f"Bond({atom_str}): value={self.value:.4f} ({actual_value:.4f} Å), "
                f"range=[{self.min_values[0]:.4f}, {self.max_values[0]:.4f}] Å, "
                f"moving_atoms={[idx + 1 for idx in self.moving_atoms]}")

    def compute(self, molecule: Molecule):
        if len(self.atom_indices) != 2:
            raise ValueError("Bond coordinate requires exactly 2 atoms")
        atom1, atom2 = molecule.atoms[self.atom_indices[0]], molecule.atoms[self.atom_indices[1]]
        bond_length = atom1.distance_to(atom2)
        self.real_values = [bond_length]
        self.value = (bond_length - self.min_values[0]) / (self.max_values[0] - self.min_values[0])
        return self

    def set(self, molecule: Molecule, target_value: float):
        if not self.moving_atoms:
            self.moving_atoms = self.get_moving_atoms(molecule, (self.atom_indices[0], self.atom_indices[1]))
        
        # Store original positions for verification
        original_positions = {idx: np.array(molecule.atoms[idx].position) for idx in range(len(molecule.atoms))}

        current_dist = molecule.atoms[self.atom_indices[0]].distance_to(molecule.atoms[self.atom_indices[1]])
        target_dist = self.get_actual_value(target_value)

        diff = target_dist - current_dist
        if abs(diff) < 1e-8:
            return

        p1 = np.array(molecule.atoms[self.atom_indices[0]].position)
        p2 = np.array(molecule.atoms[self.atom_indices[1]].position)

        vec = p2 - p1
        vec_norm = vec / np.linalg.norm(vec)
        translation = vec_norm * diff

        for idx in self.moving_atoms:
            atom = molecule.atoms[idx]
            atom.x += translation[0]
            atom.y += translation[1]
            atom.z += translation[2]

        # Update the value from the actual bond length
        self.compute(molecule)


class InternalAngleCoordinate(InternalCoordinateBase):
    type: Literal[InternalCoordinateType.ANGLE] = InternalCoordinateType.ANGLE
    @classmethod
    def initialize(cls, atom1: int, atom2: int, atom3: int, min_value: float, max_value: float, molecule: Molecule = None):
        moving_atoms = []
        if molecule:
            # Angle is 1-2-3. We move atoms attached to 3 (and 3 itself).
            moving_atoms = cls.get_moving_atoms(molecule, (atom2, atom3))
        return cls(type=InternalCoordinateType.ANGLE, atom_indices=[atom1, atom2, atom3],
                   min_values=[min_value], max_values=[max_value],
                   value=0.5, moving_atoms=moving_atoms)

    def __str__(self):
        atom_str = '-'.join(str(idx + 1) for idx in self.atom_indices)
        actual_value = self.get_actual_value(self.value)
        return (f"Angle({atom_str}): value={self.value:.4f} ({actual_value:.4f}°), "
                f"range=[{self.min_values[0]:.4f}, {self.max_values[0]:.4f}]°, "
                f"moving_atoms={[idx + 1 for idx in self.moving_atoms]}")

    def compute(self, molecule: Molecule):
        p1 = np.array(molecule.atoms[self.atom_indices[0]].position)
        p2 = np.array(molecule.atoms[self.atom_indices[1]].position)
        p3 = np.array(molecule.atoms[self.atom_indices[2]].position)
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
        self.real_values = [angle]
        self.value = (angle - self.min_values[0]) / (self.max_values[0] - self.min_values[0])
        return self

    def set(self, molecule: Molecule, target_value: float):
        if not self.moving_atoms:
             self.moving_atoms = self.get_moving_atoms(molecule, (self.atom_indices[1], self.atom_indices[2]))

        # Store original positions for verification
        original_positions = {idx: np.array(molecule.atoms[idx].position) for idx in range(len(molecule.atoms))}

        p1 = np.array(molecule.atoms[self.atom_indices[0]].position)
        p2 = np.array(molecule.atoms[self.atom_indices[1]].position)
        p3 = np.array(molecule.atoms[self.atom_indices[2]].position)

        v1 = p1 - p2
        v2 = p3 - p2

        current_angle = np.degrees(
            np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
        target_angle = self.get_actual_value(target_value)
        diff = target_angle - current_angle

        if abs(diff) < 1e-8:
            return

        # Normal to plane 1-2-3
        axis = np.cross(v1, v2)
        axis_norm = axis / np.linalg.norm(axis)

        # Rotate moving_atoms around p2 with axis axis_norm
        rot_matrix = self._get_rotation_matrix(axis_norm, np.radians(diff))

        for idx in self.moving_atoms:
            atom = molecule.atoms[idx]
            p = np.array(atom.position) - p2
            p_rot = np.dot(rot_matrix, p)
            atom.position = (p_rot + p2).tolist()

        # Verify non-moving atoms haven't changed
        self._check_non_moving_atoms_unchanged(molecule, original_positions)

        # Update the value from the actual angle
        self.compute(molecule)

    @staticmethod
    def _get_rotation_matrix(axis, theta):
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

class InternalDihedralCoordinate(InternalCoordinateBase):
    type: Literal[InternalCoordinateType.DIHEDRAL] = InternalCoordinateType.DIHEDRAL
    bond_type: InternalCoordinateBondType = InternalCoordinateBondType.UNKNOWN
    @classmethod
    def initialize(cls, atom1: int, atom2: int, atom3: int, atom4: int, min_value: float, max_value: float, molecule: Molecule = None):
        moving_atoms = []
        if molecule:
            # Dihedral is 1-2-3-4. Axis is 2-3. We move atoms attached to 3 (and 3 is fixed, so atoms beyond 3).
            # Wait, if we rotate around 2-3, atom 3 and 2 should stay. 4 and its friends move.
            moving_atoms = cls.get_moving_atoms(molecule, (atom2, atom3), collision_set=[atom1])
            # But atom 3 itself shouldn't move if we want to change the dihedral by rotating around 2-3?
            # Actually, if we rotate EVERYTHING on 3's side around the 2-3 axis, 3 will stay on the axis.
            # So moving_atoms should include 3? Yes, but since 3 is on the axis, it won't move.
            pass
        return cls(type=InternalCoordinateType.DIHEDRAL, atom_indices=[atom1, atom2, atom3, atom4],
                   min_values=[min_value], max_values=[max_value],
                   value=0.5, moving_atoms=moving_atoms)

    def __str__(self):
        atom_str = '-'.join(str(idx + 1) for idx in self.atom_indices)
        actual_value = self.get_actual_value(self.value)
        return (f"Dihedral({atom_str}): value={self.value:.4f} ({actual_value:.4f}°), "
                f"range=[{self.min_values[0]:.4f}, {self.max_values[0]:.4f}]°, "
                f"moving_atoms={[idx + 1 for idx in self.moving_atoms]}")

    def compute(self, molecule: Molecule):
        p1 = np.array(molecule.atoms[self.atom_indices[0]].position)
        p2 = np.array(molecule.atoms[self.atom_indices[1]].position)
        p3 = np.array(molecule.atoms[self.atom_indices[2]].position)
        p4 = np.array(molecule.atoms[self.atom_indices[3]].position)
        
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        n1 = np.cross(b1, b2)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(b2, b3)
        n2 /= np.linalg.norm(n2)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        
        dihedral = np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))
        self.real_values = [dihedral]
        self.value = (dihedral - self.min_values[0]) / (self.max_values[0] - self.min_values[0])
        return self

    def set(self, molecule: Molecule, target_value: float):
        if not self.moving_atoms:
            self.moving_atoms = self.get_moving_atoms(molecule, (self.atom_indices[1], self.atom_indices[2]))

        # Store original positions for verification
        original_positions = {idx: np.array(molecule.atoms[idx].position) for idx in range(len(molecule.atoms))}

        p1 = np.array(molecule.atoms[self.atom_indices[0]].position)
        p2 = np.array(molecule.atoms[self.atom_indices[1]].position)
        p3 = np.array(molecule.atoms[self.atom_indices[2]].position)
        p4 = np.array(molecule.atoms[self.atom_indices[3]].position)

        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        n1 = np.cross(b1, b2)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(b2, b3)
        n2 /= np.linalg.norm(n2)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))

        current_dihedral = np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))
        target_dihedral = self.get_actual_value(target_value)
        diff = target_dihedral - current_dihedral
        # print(f"Current dihedral: {current_dihedral:.2f}°, Target dihedral: {target_dihedral:.2f}° Diff: {diff:.2f}° target_value: {target_value:.2f} ")

        if abs(diff) < 1e-8:
            return

        axis = b2 / np.linalg.norm(b2)

        rot_matrix = InternalAngleCoordinate._get_rotation_matrix(axis, np.radians(diff))

        for idx in self.moving_atoms:
            atom = molecule.atoms[idx]
            p = np.array(atom.position) - p3  # p3 is on the axis
            p_rot = np.dot(rot_matrix, p)
            old_position = atom.position
            atom.position = (p_rot + p3).tolist()
            # print(f"Atom {idx + 1} moved from {old_position} to {atom.position}")

        # Verify non-moving atoms haven't changed
        self._check_non_moving_atoms_unchanged(molecule, original_positions)

        # Update the value from the actual dihedral
        self.compute(molecule)


class InternalFragmentAtomCoordinate(InternalCoordinateBase):
    type: Literal[InternalCoordinateType.FRAGMENT_ATOM] = InternalCoordinateType.FRAGMENT_ATOM
    # atom_indices[0]: atom to be positioned (the "atom within the fragment")
    # atom_indices[1]: reference atom 2 (root)
    # atom_indices[2]: reference atom 3 (direction)
    # atom_indices[3]: reference atom 4 (plane)

    def get_coords(self, molecule: Molecule) -> List[float]:
        """
        Calculates internal coordinates for atom 1 relative to atoms 2, 3 and 4.
        - displacement: along unit vector v23 (connecting atoms 2 and 3)
        - angle: with respect to v23
        - plane_angle: with respect to the cross product of v23 and v24
        """
        p1 = np.array(molecule.atoms[self.atom_indices[0]].position)
        p2 = np.array(molecule.atoms[self.atom_indices[1]].position)
        p3 = np.array(molecule.atoms[self.atom_indices[2]].position)
        p4 = np.array(molecule.atoms[self.atom_indices[3]].position)

        v23 = p3 - p2
        dist23 = np.linalg.norm(v23)
        u23 = v23 / dist23

        v24 = p4 - p2
        u_cp = np.cross(u23, v24)
        u_cp /= np.linalg.norm(u_cp)

        v21 = p1 - p2
        
        # displacement along u23
        displacement = np.dot(v21, u23)
        
        # angle with respect to u23
        angle = self.get_angle(p3, p2, p1)
        
        # angle with respect to cross product of 2 & 3 (and 4)
        # Assuming "cross product of 2 & 3" refers to the normal of the plane defined by 2,3,4
        # and we want the angle of v21 with respect to this normal.
        cos_phi = np.dot(v21, u_cp) / (np.linalg.norm(v21) + 1e-12)
        plane_angle = np.degrees(np.arccos(np.clip(cos_phi, -1.0, 1.0)))
        
        return [float(displacement), float(angle), float(plane_angle)]

    def set_coords(self, molecule: Molecule, coords: List[float]):
        """
        Sets the position of atom 1 based on internal coordinates.
        coords: [displacement, angle, plane_angle]
        """
        displacement, angle, plane_angle = coords
        p2 = np.array(molecule.atoms[self.atom_indices[1]].position)
        p3 = np.array(molecule.atoms[self.atom_indices[2]].position)
        p4 = np.array(molecule.atoms[self.atom_indices[3]].position)

        v23 = p3 - p2
        u23 = v23 / np.linalg.norm(v23)

        v24 = p4 - p2
        u_cp = np.cross(u23, v24)
        u_cp /= np.linalg.norm(u_cp)

        # Vector in the plane (u23, u_cp) perpendicular to u23
        u_perp = np.cross(u_cp, u23)
        u_perp /= np.linalg.norm(u_perp)

        # We need a vector v such that:
        # v . u23 = cos(angle) * |v|
        # v . u_cp = cos(plane_angle) * |v|
        
        # Let v be the direction vector of v21 (normalized).
        # v = a*u23 + b*u_perp + c*u_cp
        # v . u23 = a = cos(angle)
        # v . u_cp = c = cos(plane_angle)
        # a^2 + b^2 + c^2 = 1 => b^2 = 1 - a^2 - c^2
        
        a = np.cos(np.radians(angle))
        c = np.cos(np.radians(plane_angle))
        b2 = 1.0 - a**2 - c**2
        if b2 < 0:
            # Numerically it might be slightly negative
            b = 0.0
        else:
            b = np.sqrt(b2)
            
        v_dir = a * u23 + b * u_perp + c * u_cp
        
        # We need the absolute distance to place the atom correctly.
        # displacement = |v21| * cos(angle) => |v21| = displacement / cos(angle)
        if abs(a) > 1e-8:
            dist = displacement / a
        else:
            # If angle is 90 degrees, displacement must be 0.
            # We need another way to get dist. 
            # In this case displacement is not enough to fix distance.
            # But the prompt says displacement is one of the coordinates.
            # "displacement along the unit vector ... an angle ... and an angle"
            # If angle is 90, displacement is always 0 regardless of distance.
            # This suggests the coordinates might be (displacement, dist_perp, azimuthal_angle) 
            # or something else.
            # "displacement along ... an angle ... and an angle"
            # Let's assume it's (displacement, angle, plane_angle) and handle a=0 as an edge case.
            # Actually, if we use (dist, angle, plane_angle), it's more robust.
            # But the prompt said displacement.
            dist = 1.0 # Default if we can't determine it? 
            # Let's re-read carefully: "displacement along the unit vector ... an angle ... and an angle"
            # Maybe it means (x, angle1, angle2) where x is displacement.
            # If x = r * cos(theta), then r = x / cos(theta).
            pass

        # If we use the projection of p1-p2 onto the plane normal to u23, 
        # that would be another coordinate.
        
        # Let's try another interpretation:
        # r = p1 - p2
        # r = x*u23 + y*u_perp + z*u_cp
        # x = displacement (given)
        # angle(r, u23) = theta => x = |r| cos(theta) => |r| = x / cos(theta)
        # angle(r, u_cp) = phi => z = |r| cos(phi)
        # Then y is determined by |r|^2 = x^2 + y^2 + z^2
        
        if abs(a) > 1e-8:
            r_mag = displacement / a
            z = r_mag * c
            y2 = r_mag**2 - displacement**2 - z**2
            y = np.sqrt(max(0, y2))
            molecule.atoms[self.atom_indices[0]].position = (p2 + displacement * u23 + y * u_perp + z * u_cp).tolist()


    def compute(self, molecule: Molecule):
        coords = self.get_coords(molecule)
        self.real_values = coords
        # For 'value' (normalized [0,1]), we might need a way to combine them or pick one.
        # But if we follow InternalFragmentCoordinate, it uses mean of diffs.
        # However, InternalFragmentAtomCoordinate doesn't seem to have min/max for each coord easily available in compute if not stored.
        # Looking at InternalCoordinateBase, min_values and max_values ARE lists.
        # So we can compute normalized values for each.
        
        diffs = []
        for i in range(len(coords)):
            if i < len(self.min_values) and i < len(self.max_values):
                span = self.max_values[i] - self.min_values[i]
                if abs(span) > 1e-12:
                    diffs.append((coords[i] - self.min_values[i]) / span)
                else:
                    diffs.append(0.0)
        
        self.value = np.mean(diffs) if diffs else 0.0
        return self

class InternalFragmentCoordinate(InternalCoordinateBase):
    type: Literal[InternalCoordinateType.FRAGMENT] = InternalCoordinateType.FRAGMENT
    fragment_indices: List[int] = []
    bonds: List[InternalBondCoordinate] = []
    angles: List[InternalAngleCoordinate] = []
    dihedrals: List[InternalDihedralCoordinate] = []

    def __str__(self):
        fragment_str = ','.join(str(idx + 1) for idx in self.fragment_indices)
        lines = [f"Fragment({fragment_str}): value={self.value:.4f}"]
        if self.bonds:
            lines.append(f"  Bonds ({len(self.bonds)}):")
            for b in self.bonds:
                lines.append(f"    {b}")
        if self.angles:
            lines.append(f"  Angles ({len(self.angles)}):")
            for a in self.angles:
                lines.append(f"    {a}")
        if self.dihedrals:
            lines.append(f"  Dihedrals ({len(self.dihedrals)}):")
            for d in self.dihedrals:
                lines.append(f"    {d}")
        return '\n'.join(lines)

    @classmethod
    def initialize(cls, indices: List[int], mol1: Molecule, mol2: Molecule, debug: bool = False):

        # Remove duplicates but preserve order (first atom should not move)
        seen = set()
        fragment_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                fragment_indices.append(idx)

        # Validate that all indices are present in both molecules
        num_atoms_mol1 = len(mol1.atoms)
        num_atoms_mol2 = len(mol2.atoms)
        for idx in fragment_indices:
            if idx < 0 or idx >= num_atoms_mol1:
                raise ValueError(f"Index {idx} is out of range for mol1 (has {num_atoms_mol1} atoms)")
            if idx < 0 or idx >= num_atoms_mol2:
                raise ValueError(f"Index {idx} is out of range for mol2 (has {num_atoms_mol2} atoms)")
    
        adj1 = cls._get_adjacency(mol1)
        adj2 = cls._get_adjacency(mol2)

        internal_bonds_indices = []
        internal_angles_indices = []
        internal_dihedrals_indices = []

        # 1. Internal bonds, angles, dihedrals (using mol1 for topology)
        for i in range(len(fragment_indices)):
            idx_i = fragment_indices[i]
            for neighbor in adj1[idx_i]:
                if neighbor in fragment_indices:
                    j = fragment_indices.index(neighbor)
                    if j > i:
                        internal_bonds_indices.append([idx_i, neighbor])
                        # Angles
                        for neighbor_j in adj1[neighbor]:
                            if neighbor_j != idx_i and neighbor_j in fragment_indices:
                                k = fragment_indices.index(neighbor_j)
                                if k > j and [idx_i, neighbor, neighbor_j] not in internal_angles_indices:
                                    internal_angles_indices.append([idx_i, neighbor, neighbor_j])
                                # Dihedrals
                                for neighbor_k in adj1[neighbor_j]:
                                    if neighbor_k != neighbor and neighbor_k in fragment_indices:
                                        l = fragment_indices.index(neighbor_k)
                                        if l > k and [idx_i, neighbor, neighbor_j,
                                                      neighbor_k] not in internal_dihedrals_indices:
                                            internal_dihedrals_indices.append([idx_i, neighbor, neighbor_j, neighbor_k])

        def get_moving_atoms_union(molecule1, molecule2, bond, collision_set=None):
            moving1 = set(cls.get_moving_atoms(molecule1, bond, collision_set))
            moving2 = set(cls.get_moving_atoms(molecule2, bond, collision_set))
            return sorted(list(moving1.union(moving2)))

        # Create Coordinate objects
        bonds = []
        for b in internal_bonds_indices:
            min_val = mol1.atoms[b[0]].distance_to(mol1.atoms[b[1]])
            max_val = mol2.atoms[b[0]].distance_to(mol2.atoms[b[1]])
            moving = get_moving_atoms_union(mol1, mol2, (b[0], b[1]), collision_set=b)
            bonds.append(InternalBondCoordinate(
                type=InternalCoordinateType.BOND,
                atom_indices=b,
                min_values=[min_val],
                max_values=[max_val],
                moving_atoms=moving
            ))
        bonds = []
        angles = []
        for a in internal_angles_indices:
            p1_1 = np.array(mol1.atoms[a[0]].position)
            p2_1 = np.array(mol1.atoms[a[1]].position)
            p3_1 = np.array(mol1.atoms[a[2]].position)
            min_val = cls.get_angle(p1_1, p2_1, p3_1)

            p1_2 = np.array(mol2.atoms[a[0]].position)
            p2_2 = np.array(mol2.atoms[a[1]].position)
            p3_2 = np.array(mol2.atoms[a[2]].position)
            max_val = cls.get_angle(p1_2, p2_2, p3_2)

            moving = get_moving_atoms_union(mol1, mol2, (a[1], a[2]), collision_set=a)
            angles.append(InternalAngleCoordinate(
                type=InternalCoordinateType.ANGLE,
                atom_indices=a,
                min_values=[min_val],
                max_values=[max_val],
                moving_atoms=moving
            ))
        angles = []
        dihedrals = []
        for d in internal_dihedrals_indices:
            p1_1 = np.array(mol1.atoms[d[0]].position)
            p2_1 = np.array(mol1.atoms[d[1]].position)
            p3_1 = np.array(mol1.atoms[d[2]].position)
            p4_1 = np.array(mol1.atoms[d[3]].position)
            min_val = cls.get_dihedral(p1_1, p2_1, p3_1, p4_1)

            p1_2 = np.array(mol2.atoms[d[0]].position)
            p2_2 = np.array(mol2.atoms[d[1]].position)
            p3_2 = np.array(mol2.atoms[d[2]].position)
            p4_2 = np.array(mol2.atoms[d[3]].position)
            max_val = cls.get_dihedral(p1_2, p2_2, p3_2, p4_2)

            moving = get_moving_atoms_union(mol1, mol2, (d[1], d[2]), collision_set=[d[0], d[1], d[2]])
            dihedrals.append(InternalDihedralCoordinate(
                type=InternalCoordinateType.DIHEDRAL,
                atom_indices=d,
                min_values=[min_val],
                max_values=[max_val],
                moving_atoms=moving
            ))

        if debug:
            print(f"Selected Internal Coordinates for Fragment {indices}:")
            print(f"\nBond values:")
            for b in bonds:
                print(f"  Bond {[idx + 1 for idx in b.atom_indices]}: min={b.min_values[0]:.4f}, max={b.max_values[0]:.4f}")

            print(f"\nAngle values:")
            for a in angles:
                print(f"  Angle {[idx + 1 for idx in a.atom_indices]}: min={a.min_values[0]:.4f}, max={a.max_values[0]:.4f}")

            print(f"\nDihedral values:")
            for d in dihedrals:
                print(f"  Dihedral {[idx + 1 for idx in d.atom_indices]}: min={d.min_values[0]:.4f}, max={d.max_values[0]:.4f}")

        involved_atoms = set(fragment_indices)
        for b in bonds: involved_atoms.update(b.atom_indices)
        for a in angles: involved_atoms.update(a.atom_indices)
        for d in dihedrals: involved_atoms.update(d.atom_indices)
        
        involved_atoms_list = sorted(list(involved_atoms))
        
        return cls(type=InternalCoordinateType.FRAGMENT, 
                   atom_indices=involved_atoms_list, 
                   fragment_indices=fragment_indices,
                   bonds=bonds,
                   angles=angles,
                   dihedrals=dihedrals,
                   min_values=[], max_values=[], # Not used anymore
                   value=0.0)

    def compute(self, molecule: Molecule):
        real_vals = []
        diffs = []
        for b in self.bonds:
            b.compute(molecule)
            diffs.append(b.value)
            real_vals.extend(b.real_values)
        for a in self.angles:
            a.compute(molecule)
            diffs.append(a.value)
            real_vals.extend(a.real_values)
        for d in self.dihedrals:
            d.compute(molecule)
            diffs.append(d.value)
            real_vals.extend(d.real_values)
        
        self.real_values = real_vals
        self.value = np.mean(diffs) if diffs else 0.0
        return self

    def set(self, molecule: Molecule, target_value: float):
        # To avoid redundant moves and cycles within the fragment, 
        # we build a spanning tree and only set coordinates that belong to this tree.
        # This ensures a deterministic update of the fragment internal structure.
        
        adj = self._get_adjacency(molecule)
        root = self.fragment_indices[0]
        tree_edges = [] # (parent, child)
        visited = {root}
        queue = [root]
        
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor in self.fragment_indices and neighbor not in visited:
                    visited.add(neighbor)
                    tree_edges.append((curr, neighbor))
                    queue.append(neighbor)

        # 1. Update tree bonds
        for parent, child in tree_edges:
            # Find the bond object
            for b in self.bonds:
                if (b.atom_indices[0] == parent and b.atom_indices[1] == child) or (b.atom_indices[0] == child and b.atom_indices[1] == parent):
                    b.set(molecule, target_value)
                    break

        # 2. Update angles between tree edges
        # For each tree edge (parent, child), if child has other children in the tree,
        # update the angle parent-child-next_child.
        for parent, child in tree_edges:
            curr = child
            p_of_curr = parent
            # next_child must be a child of 'curr' in the tree
            for next_child in adj[curr]:
                if next_child in visited and (curr, next_child) in tree_edges:
                    # Angle p_of_curr - curr - next_child
                    for a in self.angles:
                        if a.atom_indices[1] == curr and ((a.atom_indices[0] == p_of_curr and a.atom_indices[2] == next_child) or (a.atom_indices[0] == next_child and a.atom_indices[2] == p_of_curr)):
                            a.set(molecule, target_value)
                            break

        # 3. Update dihedrals
        # We update all stored dihedrals.
        # print("BEFORE ROT: ", molecule.atoms[8].position)
        for d in self.dihedrals:
            d.set(molecule, target_value)
        # print("After: ", molecule.atoms[8].position)
        self.compute(molecule)


class Anchor(EmbeddedModel):
    atom_idx: int  # The outside atom index
    fragment_root_idx: int  # The atom in the fragment it's connected to
    neighbor_indices: List[int]  # 2 neighbors in the fragment to define the local coordinate system
    moving_atoms: List[int]
    min_values: List[float]  # [distance, angle, plane_angle]
    max_values: List[float]  # [distance, angle, plane_angle]
    # Store initial local coordinate system to compute rotation
    # v1, v2, v3 for mol1
    initial_local_system: List[List[float]] = [] # [v1, v2, v3]

    def set(self, molecule: Molecule, target_value: float, fragment_positions: Dict[int, np.ndarray]):
        """
        Interpolates the anchor atom position and orientation based on fragment positions.
        fragment_positions: Current positions of fragment atoms.
        """
        d = self.min_values[0] + target_value * (self.max_values[0] - self.min_values[0])
        ang = self.min_values[1] + target_value * (self.max_values[1] - self.min_values[1])
        pang = self.min_values[2] + target_value * (self.max_values[2] - self.min_values[2])

        p_root = fragment_positions[self.fragment_root_idx]
        p_n1 = fragment_positions[self.neighbor_indices[0]]
        p_n2 = fragment_positions[self.neighbor_indices[1]]

        # Define current local coordinate system
        v1 = p_n1 - p_root
        v2 = p_n2 - p_root
        v1 /= np.linalg.norm(v1)
        
        # Orthogonalize v2
        v2 = v2 - np.dot(v2, v1) * v1
        v2 /= np.linalg.norm(v2)
        
        # Normal to plane
        v3 = np.cross(v1, v2)
        v3 /= np.linalg.norm(v3)

        rad_ang = np.radians(ang)
        rad_pang = np.radians(pang)

        # Direction vector in current local system:
        # Rotate v1 towards v2 by rad_ang
        dir_vec = np.cos(rad_ang) * v1 + np.sin(rad_ang) * v2
        # Now rotate this vector towards v3 by rad_pang
        final_dir = np.cos(rad_pang) * dir_vec + np.sin(rad_pang) * v3
        
        new_pos = p_root + d * final_dir
        
        # Rotation calculation:
        # We want to rotate from the initial local system to the current one.
        # initial_local_system = [v1_0, v2_0, v3_0]
        # current_local_system = [v1, v2, v3] (already computed)
        
        if self.initial_local_system:
            v1_0 = np.array(self.initial_local_system[0])
            v2_0 = np.array(self.initial_local_system[1])
            v3_0 = np.array(self.initial_local_system[2])
            
            # Rotation R1 maps (v1_0, v2_0, v3_0) to (v1, v2, v3)
            # R1 = [v1, v2, v3] @ [v1_0, v2_0, v3_0].T
            R1 = np.column_stack([v1, v2, v3]) @ np.column_stack([v1_0, v2_0, v3_0]).T
            
            # Initial anchor-root vector direction in world coordinates:
            ang0 = self.min_values[1]
            pang0 = self.min_values[2]
            rad_ang0 = np.radians(ang0)
            rad_pang0 = np.radians(pang0)
            dir_vec0 = np.cos(rad_ang0) * v1_0 + np.sin(rad_ang0) * v2_0
            final_dir0 = np.cos(rad_pang0) * dir_vec0 + np.sin(rad_pang0) * v3_0
            
            # In the new basis (v1, v2, v3), the "original" anchor direction would be R1 @ final_dir0.
            # But the interpolated direction is final_dir.
            v_start = R1 @ final_dir0
            v_end = final_dir
            
            axis = np.cross(v_start, v_end)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                axis /= norm
                cos_theta = np.dot(v_start, v_end)
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                R2 = InternalAngleCoordinate._get_rotation_matrix(axis, theta)
            else:
                if np.dot(v_start, v_end) < -0.999999: # Almost opposite
                    axis = np.cross(v_start, [1, 0, 0])
                    if np.linalg.norm(axis) < 1e-8:
                        axis = np.cross(v_start, [0, 1, 0])
                    axis /= np.linalg.norm(axis)
                    R2 = InternalAngleCoordinate._get_rotation_matrix(axis, np.pi)
                else:
                    R2 = np.eye(3)
            
            R_total = R2 @ R1
        else:
            R_total = np.eye(3)

        # Move anchor and all its moving atoms
        old_anchor_pos = np.array(molecule.atoms[self.atom_idx].position)
        
        # Determine the initial anchor orientation before update.
        # This is the current world position of anchor and its children.
        # Note: at the start of set(), the fragment was already moved. 
        # But moving_atoms (children) are still at their original world positions relative to the OLD anchor position.
        
        for idx in self.moving_atoms:
            atom = molecule.atoms[idx]
            p = np.array(atom.position)
            # 1. Translate such that the anchor is at the origin
            p -= old_anchor_pos
            # 2. Apply total rotation
            p = R_total @ p
            # 3. Translate to the new anchor position
            p += new_pos
            atom.position = p.tolist()

    def __str__(self):
        return f"Anchor({self.atom_idx}, {self.fragment_root_idx}, {self.neighbor_indices}, {self.moving_atoms})"

class InternalFragmentInterpolator(InternalCoordinateBase):
    type: Literal[InternalCoordinateType.INTERPOLATOR] = InternalCoordinateType.INTERPOLATOR
    fragment_indices: List[int]
    min_coords: Dict[int, List[float]]  # atom_idx -> [x, y, z] for mol1
    max_coords: Dict[int, List[float]]  # atom_idx -> [x, y, z] for mol2 (aligned)
    anchors: List[Anchor]

    def __str__(self):
        fragment_str = ','.join(str(idx + 1) for idx in self.fragment_indices)
        lines = [f"FragmentInterpolator({fragment_str}): value={self.value:.4f}"]
        lines.append(f"  Fragment atoms: {len(self.fragment_indices)}")
        if self.anchors:
            lines.append(f"  Anchors ({len(self.anchors)}):")
            for i, anchor in enumerate(self.anchors):
                lines.append(
                    f"    Anchor {i}: atom {anchor.atom_idx + 1} -> fragment_root {anchor.fragment_root_idx + 1}, "
                    f"neighbors={[idx + 1 for idx in anchor.neighbor_indices]}, "
                    f"moving_atoms={[idx + 1 for idx in anchor.moving_atoms]}")
        return '\n'.join(lines)

    @classmethod
    def initialize(cls, indices: List[int], mol1: Molecule, mol2: Molecule):
        # 1. Alignment
        # Extract fragment positions
        p1 = np.array([mol1.atoms[idx].position for idx in indices])
        p2 = np.array([mol2.atoms[idx].position for idx in indices])
        
        # Kabsch alignment: align p2 to p1
        p2_aligned = cls._align_points(p1, p2)
        
        min_coords = {idx: mol1.atoms[idx].position for idx in indices}
        max_coords = {idx: p2_aligned[i].tolist() for i, idx in enumerate(indices)}
        
        # 2. Identify Anchors
        adj1 = cls._get_adjacency(mol1)
        adj2 = cls._get_adjacency(mol2)
        
        anchors = []
        fragment_set = set(indices)
        
        for root_idx in indices:
            # Atoms connected to fragment but not in fragment
            for neighbor in adj1[root_idx]:
                if neighbor not in fragment_set:
                    # Found an anchor atom
                    anchor_idx = neighbor
                    
                    # Need 2 neighbors in fragment for root_idx
                    frag_neighbors = [n for n in adj1[root_idx] if n in fragment_set]
                    if len(frag_neighbors) < 2:
                        # Fallback: search neighbors of neighbors in fragment
                        for fn in frag_neighbors:
                            frag_neighbors.extend([n for n in adj1[fn] if n in fragment_set and n != root_idx])
                        frag_neighbors = sorted(list(set(frag_neighbors)))
                    
                    if len(frag_neighbors) < 2:
                         # Still not enough? Just pick any other fragment atoms
                         other_frag = [idx for idx in indices if idx != root_idx]
                         frag_neighbors.extend(other_frag[:2-len(frag_neighbors)])
                    
                    n1_idx, n2_idx = frag_neighbors[0], frag_neighbors[1]
                    
                    # Compute values for mol1
                    vals1 = cls._compute_anchor_values(mol1, anchor_idx, root_idx, n1_idx, n2_idx)
                    
                    # Compute values for mol2
                    # Note: for mol2 we use the same indices.
                    vals2 = cls._compute_anchor_values(mol2, anchor_idx, root_idx, n1_idx, n2_idx)

                    # Compute initial local system for mol1
                    p_r = np.array(mol1.atoms[root_idx].position)
                    p_n1 = np.array(mol1.atoms[n1_idx].position)
                    p_n2 = np.array(mol1.atoms[n2_idx].position)
                    v1_0 = p_n1 - p_r
                    v2_0 = p_n2 - p_r
                    v1_0 /= np.linalg.norm(v1_0)
                    v2_0 = v2_0 - np.dot(v2_0, v1_0) * v1_0
                    v2_0 /= np.linalg.norm(v2_0)
                    v3_0 = np.cross(v1_0, v2_0)
                    v3_0 /= np.linalg.norm(v3_0)
                    
                    # moving_atoms for anchor: all atoms reachable from anchor_idx without going through root_idx
                    moving1 = set(cls.get_moving_atoms(mol1, (root_idx, anchor_idx)))
                    moving2 = set(cls.get_moving_atoms(mol2, (root_idx, anchor_idx)))
                    moving_union = sorted(list(moving1.union(moving2)))
                    
                    anchors.append(Anchor(
                        atom_idx=anchor_idx,
                        fragment_root_idx=root_idx,
                        neighbor_indices=[n1_idx, n2_idx],
                        moving_atoms=moving_union,
                        min_values=vals1,
                        max_values=vals2,
                        initial_local_system=[v1_0.tolist(), v2_0.tolist(), v3_0.tolist()]
                    ))
        
        return cls(
            type=InternalCoordinateType.INTERPOLATOR,
            atom_indices=indices, # Primary indices are the fragment
            fragment_indices=indices,
            min_coords=min_coords,
            max_coords=max_coords,
            anchors=anchors,
            min_values=[], max_values=[]
        )

    @staticmethod
    def _align_points(p1, p2):
        """Aligns p2 to p1 using Kabsch algorithm."""
        # Center of mass
        c1 = np.mean(p1, axis=0)
        c2 = np.mean(p2, axis=0)
        p1c = p1 - c1
        p2c = p2 - c2
        
        # Covariance matrix
        H = np.dot(p2c.T, p1c)
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T
        
        # Rotation matrix
        R = np.dot(V, U.T)
        
        # Reflection check
        if np.linalg.det(R) < 0:
            V[:, -1] *= -1
            R = np.dot(V, U.T)
            
        p2_aligned = np.dot(p2c, R.T) + c1
        return p2_aligned

    @classmethod
    def _compute_anchor_values(cls, mol: Molecule, anchor_idx: int, root_idx: int, n1_idx: int, n2_idx: int):
        p_a = np.array(mol.atoms[anchor_idx].position)
        p_r = np.array(mol.atoms[root_idx].position)
        p_n1 = np.array(mol.atoms[n1_idx].position)
        p_n2 = np.array(mol.atoms[n2_idx].position)
        
        dist = np.linalg.norm(p_a - p_r)
        
        # angle n1-root-anchor
        v_ra = p_a - p_r
        v_rn1 = p_n1 - p_r
        angle = cls.get_angle(p_n1, p_r, p_a)
        
        # plane angle: angle between v_ra and plane (root, n1, n2)
        v_rn2 = p_n2 - p_r
        normal = np.cross(v_rn1, v_rn2)
        normal /= np.linalg.norm(normal)
        
        # angle between vector and normal
        cos_beta = np.dot(v_ra, normal) / (np.linalg.norm(v_ra))
        # angle between vector and plane is 90 - beta
        # sin(plane_angle) = cos(beta)
        plane_angle = np.degrees(np.arcsin(np.clip(cos_beta, -1.0, 1.0)))
        
        # Wait, if we use the spherical coordinates in set(), we need the angle in the plane too.
        # Let's redefine:
        # v1 = v_rn1 normalized
        # v3 = normal
        # v2 = v3 x v1
        # v_ra_dir = v_ra / dist
        # v_ra_dir = cos(pang) * (cos(ang) * v1 + sin(ang) * v2) + sin(pang) * v3
        
        v1 = v_rn1 / np.linalg.norm(v_rn1)
        v3 = normal
        v2 = np.cross(v3, v1)
        
        v_ra_dir = v_ra / dist
        
        sin_pang = np.dot(v_ra_dir, v3)
        pang = np.degrees(np.arcsin(np.clip(sin_pang, -1.0, 1.0)))
        
        # Projection on plane
        proj = v_ra_dir - sin_pang * v3
        if np.linalg.norm(proj) > 1e-8:
            proj /= np.linalg.norm(proj)
            cos_ang = np.dot(proj, v1)
            sin_ang = np.dot(proj, v2)
            ang = np.degrees(np.arctan2(sin_ang, cos_ang))
        else:
            ang = 0.0
            
        return [float(dist), float(ang), float(pang)]

    def compute(self, molecule: Molecule):
        # We can define real_values as the flat list of all atom coordinates in the fragment
        # but that might be overkill. 
        # Perhaps more consistent is to use the current interpolation factor if it can be determined.
        # But this class is mostly for setting.
        # For now, let's just ensure it has real_values as an empty list or similar to avoid errors.
        self.real_values = []
        for idx in self.fragment_indices:
            self.real_values.extend(molecule.atoms[idx].position)
        return self

    def set(self, molecule: Molecule, target_value: float):
        # 1. Interpolate fragment
        current_fragment_positions = {}
        for idx in self.fragment_indices:
            p1 = np.array(self.min_coords[idx])
            p2 = np.array(self.max_coords[idx])
            pos = p1 + target_value * (p2 - p1)
            molecule.atoms[idx].position = pos.tolist()
            current_fragment_positions[idx] = pos
            
        # 2. Interpolate anchors
        for anchor in self.anchors:
            anchor.set(molecule, target_value, current_fragment_positions)

        self.value = target_value
        self.compute(molecule)




# Define the Union of all derived types
InternalCoordinate = Annotated[
    Union[
        InternalBondCoordinate,
        InternalAngleCoordinate,
        InternalDihedralCoordinate,
        InternalFragmentCoordinate,
        InternalFragmentInterpolator
    ],
    PydanticField(discriminator="type")
]

class InternalCoordinatesList(Model, GenericListMixin[InternalCoordinate]):
    elements: List[InternalCoordinate] = Field(default_factory=list, description="List of internal coordinates")

    @classmethod
    def from_molecule(cls, molecule: Molecule) -> "InternalCoordinatesList":
        """
        Parses the properties of the molecule to extract all internal coordinates.
        Expects molecule.properties to have a "selections" key which is a list of dictionaries.
        Each dictionary should have 'type' (bond, cleave, angle, dihedral) and 'atoms' (list of indices).
        """
        ic_list = cls()
        selections = molecule.properties.get("selections", [])
        for selection in selections:
            sel_type = selection.get('type')
            atoms = selection.get('atoms', [])
            if sel_type in ["bond", "cleave"]:
                if len(atoms) != 2:
                    continue
                a1, a2 = molecule.atoms[atoms[0]], molecule.atoms[atoms[1]]
                l1 = a1.distance_to(a2)
                if sel_type == "cleave":
                    l2 = l1 + 1.0
                else:
                    l2 = 1.1 * l1
                l1 = 0.9 * l1
                bc = InternalBondCoordinate.initialize(atoms[0], atoms[1], l1, l2, molecule)
                ic_list.elements.append(bc)
            elif sel_type == "angle":
                if len(atoms) != 3:
                    continue
                p1 = np.array(molecule.atoms[atoms[0]].position)
                p2 = np.array(molecule.atoms[atoms[1]].position)
                p3 = np.array(molecule.atoms[atoms[2]].position)
                v1 = p1 - p2
                v2 = p3 - p2
                angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
                # Default range +/- 10 degrees?
                ac = InternalAngleCoordinate.initialize(atoms[0], atoms[1], atoms[2], angle - 10.0, angle + 10.0, molecule)
                ic_list.elements.append(ac)
            elif sel_type == "dihedral":
                if len(atoms) != 4:
                    continue
                p1 = np.array(molecule.atoms[atoms[0]].position)
                p2 = np.array(molecule.atoms[atoms[1]].position)
                p3 = np.array(molecule.atoms[atoms[2]].position)
                p4 = np.array(molecule.atoms[atoms[3]].position)
                b1 = p2 - p1
                b2 = p3 - p2
                b3 = p4 - p3
                n1 = np.cross(b1, b2)
                n2 = np.cross(b2, b3)
                m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                dihedral = np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))
                # Default range +/- 20 degrees?
                dc = InternalDihedralCoordinate.initialize(atoms[0], atoms[1], atoms[2], atoms[3], dihedral - 20.0, dihedral + 20.0, molecule)
                ic_list.elements.append(dc)
        return ic_list

    @classmethod
    def from_states(cls, mol1: Molecule, mol2: Molecule) -> "InternalCoordinatesList":
        """
        Sets the min-max values of the internal coordinates based on two molecules.
        Only the first molecule (mol1) has the 'selections' property to define which coordinates to track.
        """
        ic_list = cls()
        selections = mol1.properties.get("selections", [])
        for selection in selections:
            sel_type = selection.get('type')
            atoms = selection.get('atoms', [])
            if sel_type in ["bond", "cleave"]:
                if len(atoms) != 2:
                    continue
                l1 = mol1.atoms[atoms[0]].distance_to(mol1.atoms[atoms[1]])
                l2 = mol2.atoms[atoms[0]].distance_to(mol2.atoms[atoms[1]])
                min_l = min(l1, l2)
                max_l = max(l1, l2)
                # Add some buffer? In make_internal_coordinates it was 0.9 and 1.1.
                # Here we strictly take from the two states as requested.
                bc = InternalBondCoordinate.initialize(atoms[0], atoms[1], min_l, max_l, mol1)
                ic_list.elements.append(bc)
            elif sel_type == "angle":
                if len(atoms) != 3:
                    continue
                def get_ang(m):
                    p1 = np.array(m.atoms[atoms[0]].position)
                    p2 = np.array(m.atoms[atoms[1]].position)
                    p3 = np.array(m.atoms[atoms[2]].position)
                    v1 = p1 - p2
                    v2 = p3 - p2
                    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
                
                ang1 = get_ang(mol1)
                ang2 = get_ang(mol2)
                ac = InternalAngleCoordinate.initialize(atoms[0], atoms[1], atoms[2], min(ang1, ang2), max(ang1, ang2), mol1)
                ic_list.elements.append(ac)
            elif sel_type == "dihedral":
                if len(atoms) != 4:
                    continue
                def get_dih(m):
                    p1 = np.array(m.atoms[atoms[0]].position)
                    p2 = np.array(m.atoms[atoms[1]].position)
                    p3 = np.array(m.atoms[atoms[2]].position)
                    p4 = np.array(m.atoms[atoms[3]].position)
                    b1 = p2 - p1
                    b2 = p3 - p2
                    b3 = p4 - p3
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                    return np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))
                
                dih1 = get_dih(mol1)
                dih2 = get_dih(mol2)
                
                # Dihedrals are tricky because of wrapping around 180/-180.
                diff = (dih2 - dih1 + 180) % 360 - 180
                dih2_unwrapped = dih1 + diff
                
                min_d = min(dih1, dih2_unwrapped)
                max_d = max(dih1, dih2_unwrapped)
                
                dc = InternalDihedralCoordinate.initialize(atoms[0], atoms[1], atoms[2], atoms[3], min_d, max_d, mol1)
                ic_list.elements.append(dc)
        return ic_list


def get_fragment_connection_coords(mol: Molecule, atom_indices: List[int]) -> Dict[str, float]:
    """
    Given a set of 3 or 4 atoms, extract the internal coordinates required to implement 
    the fragment moves (distance, angle, and optional dihedral).
    If 3 atoms are given (a1, b1, b2), returns dist(a1, b1), angle(a1, b1, b2).
    If 4 atoms are given (a1, a2, b1, b2), returns dist(a1, b1), angle(a1, b1, b2), dihedral(a2, a1, b1, b2).
    """
    p = [np.array(mol.atoms[idx].position) for idx in atom_indices]
    
    if len(atom_indices) == 3:
        # a1, b1, b2
        dist = np.linalg.norm(p[0] - p[1])
        angle = InternalCoordinateBase.get_angle(p[0], p[1], p[2])
        return {"distance": float(dist), "angle": float(angle)}
    elif len(atom_indices) == 4:
        # a2, a1, b1, b2
        dist = np.linalg.norm(p[1] - p[2])
        angle = InternalCoordinateBase.get_angle(p[1], p[2], p[3])
        dihedral = InternalCoordinateBase.get_dihedral(p[0], p[1], p[2], p[3])
        return {"distance": float(dist), "angle": float(angle), "dihedral": float(dihedral)}
    else:
        raise ValueError("Exactly 3 or 4 atoms must be specified.")

