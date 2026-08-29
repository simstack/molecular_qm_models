import numpy as np
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .molecule import Molecule

# Constant to decide whether angles are in degrees or radians
USE_DEGREES = True


class Bond:
    """Helper class to compute bond distance between two atoms."""
    def __init__(self, p1: Union[List[float], np.ndarray], p2: Union[List[float], np.ndarray]):
        self.p1 = np.asarray(p1, dtype=float)
        self.p2 = np.asarray(p2, dtype=float)

    def compute(self) -> float:
        """Computes the distance between two 3D coordinates."""
        return np.linalg.norm(self.p1 - self.p2)

    @classmethod
    def from_molecule(cls, molecule: "Molecule", i1: int, i2: int) -> float:
        """
        Creates a Bond instance from a Molecule and two atom indices, 
        and returns the computed distance.
        """
        atoms = molecule.atoms
        p1 = [atoms[i1].x, atoms[i1].y, atoms[i1].z]
        p2 = [atoms[i2].x, atoms[i2].y, atoms[i2].z]
        return cls(p1, p2).compute()


class Angle:
    """Helper class to compute angle between three atoms."""
    def __init__(self, p1: Union[List[float], np.ndarray], p2: Union[List[float], np.ndarray], p3: Union[List[float], np.ndarray]):
        self.p1 = np.asarray(p1, dtype=float)
        self.p2 = np.asarray(p2, dtype=float)
        self.p3 = np.asarray(p3, dtype=float)

    def compute(self) -> float:
        """Computes the angle between three 3D coordinates (at p2)."""
        v1 = self.p1 - self.p2
        v2 = self.p3 - self.p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        if USE_DEGREES:
            return float(np.degrees(angle_rad))
        return float(angle_rad)

    @classmethod
    def from_molecule(cls, molecule: "Molecule", i1: int, i2: int, i3: int) -> float:
        """
        Creates an Angle instance from a Molecule and three atom indices, 
        and returns the computed angle.
        """
        atoms = molecule.atoms
        p1 = [atoms[i1].x, atoms[i1].y, atoms[i1].z]
        p2 = [atoms[i2].x, atoms[i2].y, atoms[i2].z]
        p3 = [atoms[i3].x, atoms[i3].y, atoms[i3].z]
        return cls(p1, p2, p3).compute()


class Dihedral:
    """Helper class to compute dihedral angle between four atoms."""
    def __init__(self, p1: Union[List[float], np.ndarray], p2: Union[List[float], np.ndarray], 
                 p3: Union[List[float], np.ndarray], p4: Union[List[float], np.ndarray]):
        self.p1 = np.asarray(p1, dtype=float)
        self.p2 = np.asarray(p2, dtype=float)
        self.p3 = np.asarray(p3, dtype=float)
        self.p4 = np.asarray(p4, dtype=float)

    def compute(self) -> float:
        """Computes the dihedral angle between four 3D coordinates."""
        v1 = self.p2 - self.p1
        v2 = self.p3 - self.p2
        v3 = self.p4 - self.p3
        
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        
        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)
        
        if norm_n1 < 1e-8 or norm_n2 < 1e-8:
            return 0.0
            
        n1 /= norm_n1
        n2 /= norm_n2
        
        norm_v2 = np.linalg.norm(v2)
        if norm_v2 < 1e-8:
            return 0.0
            
        m1 = np.cross(n1, v2 / norm_v2)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        angle_rad = np.arctan2(y, x)
        
        if USE_DEGREES:
            return float(np.degrees(angle_rad))
        return float(angle_rad)

    @classmethod
    def from_molecule(cls, molecule: "Molecule", i1: int, i2: int, i3: int, i4: int) -> float:
        """
        Creates a Dihedral instance from a Molecule and four atom indices, 
        and returns the computed dihedral angle.
        """
        atoms = molecule.atoms
        p1 = [atoms[i1].x, atoms[i1].y, atoms[i1].z]
        p2 = [atoms[i2].x, atoms[i2].y, atoms[i2].z]
        p3 = [atoms[i3].x, atoms[i3].y, atoms[i3].z]
        p4 = [atoms[i4].x, atoms[i4].y, atoms[i4].z]
        return cls(p1, p2, p3, p4).compute()
