import hashlib
import math
from copy import copy
from typing import Dict, Any, List, Optional, Iterator
from odmantic import Model, Field, EmbeddedModel, ObjectId
from pydantic import model_validator
import logging
from pathlib import Path

from simstack.models import simstack_model
from simstack.models.base_lists import ObjectListMixin
from simstack.util.generate_ui_schema import generate_ui_schema
logger = logging.getLogger(__name__)


@simstack_model
class Atom(EmbeddedModel):
    field_name: str = "Atom"
    element: str
    x: float
    y: float
    z: float
    properties: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def from_atom(cls, atom: "Atom") -> "Atom":
        return Atom(element=atom.element, x=atom.x,y=atom.y,z=atom.z, properties = copy(atom.properties))

    def __iter__(self):
        """
        Allow iteration over the atomic coordinates.

        :return: An iterator over the x, y, z coordinates.
        :rtype: Iterator[float]
        """
        return iter([self.x, self.y, self.z])

    @classmethod
    def from_coords(cls,element: str, coords: List[float]):
        """
        Create an Atom object from coordinates.

        :param coords:
        :param element: The element symbol (e.g., "H", "O").
        :param coords: list of  3 coordinates
        :return: An Atom object.
        """
        return Atom(element=element, x=coords[0], y=coords[1], z=coords[2])

    @property
    def species(self):
        return self.element

    @property
    def position(self):
        return self.x, self.y, self.z

    @position.setter
    def position(self,value: List[float]):
        self.x,self.y,self.z = value

    def distance_to(self, other: "Atom") -> float:
        """
        Calculate the distance to another atom.

        :param other: Another Atom object.
        :return: The distance to the other atom.
        :rtype: float
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def make_table_entries(self,max_recursion_level=1, drop_id=True, current_level=0):
        pass

@simstack_model
class Molecule(Model):
    """
    A class representing a molecule with atoms and their coordinates.

    Attributes:
        atoms (List[Atom]): A list of Atom objects representing the atoms in the molecule.
    """
    field_name: str = "Molecule"
    atoms: List[Atom] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    smiles: Optional[str] = None
    formula: Optional[str] = None
    
    def add_atom(self, atom: Atom):
        """
        Add an atom to the molecule.

        :param atom: An Atom object to be added to the molecule.
        """
        self.atoms.append(atom)

    @classmethod
    def from_molecule(cls,molecule: "Molecule") -> "Molecule":
        new_molecule = Molecule()
        for atom in molecule.atoms:
            new_molecule.add_atom(Atom.from_atom(atom))
        new_molecule.properties = copy(molecule.properties)
        new_molecule.smiles = molecule.smiles
        new_molecule.formula = molecule.formula
        return new_molecule

    def make_smiles(self) -> str:
        try:
            from molecular_qm_util import compute_smiles
            self.smiles = compute_smiles(self)
        except ImportError:
            logger.warning("molecular_qm_util package not found. SMILES computation failed.")
            self.smiles = "Error: molecular_qm_util missing"
        return self.smiles

    def make_formula(self) -> str:
        try:
            from molecular_qm_util import compute_iupac_name
            self.formula = compute_iupac_name(self)
        except ImportError:
            logger.warning("molecular_qm_util package not found. Formula computation failed.")
            self.formula = "Error: molecular_qm_util missing"
        return self.formula

    def make_table_entries(self,**kwargs):
        field_prefix = kwargs.get("field_prefix", "")
        return {field_prefix + 'smiles': self.smiles, field_prefix + 'formula': self.formula}

    def make_column_defs_instance(self,**kwargs) -> List[Dict[str, Any]]:
        field_prefix = kwargs.get("field_prefix", "")
        return [{'headerName': 'smiles', 'field': field_prefix + 'smiles'  },
                {'headerName': 'formula', 'field': field_prefix + 'formula' }, ]

    @classmethod
    def make_column_defs(cls,**kwargs) -> List[Dict[str, Any]]:
        field_prefix = kwargs.get("field_prefix", "")
        return [{'headerName': 'smiles', 'field': field_prefix + 'smiles'},
                {'headerName': 'formula', 'field': field_prefix + 'formula'}, ]

    @classmethod
    def from_atoms(cls, atoms: List[Atom], properties: Dict[str,Any] = None) -> "Molecule":
        new_molecule = Molecule()
        for atom in atoms:
            new_molecule.add_atom(Atom.from_atom(atom))
        if properties:
            new_molecule.properties = copy(properties)
        return new_molecule

    @classmethod
    def from_sites(cls, elements: List[str], sites: List[List[float]]):
        if len(elements) != len(sites):
            logger.error("Number of elements must match number of sites.")
            raise ValueError("Number of elements must match number of sites.")

        molecule = cls()
        for element, site in zip(elements, sites):
            atom = Atom(element=element, x=site[0], y=site[1], z=site[2])
            molecule.add_atom(atom)
        return molecule

    @property
    def charge(self):
        return self.properties.get("charge", 0)

    @property
    def spin_multiplicity(self):
        return self.properties.get("spin_multiplicity", 1)

    def __iter__(self):
        """
        Iterate over the atoms in the molecule.

        :return: An iterator over the atoms in the molecule.
        :rtype: Iterator[Atom]
        """
        return iter(self.atoms)

    def __len__(self):
        """
        Get the number of atoms in the molecule.

        :return: The number of atoms in the molecule.
        :rtype: int
        """
        return len(self.atoms)


    @classmethod
    def ui_schema(cls):
            ui_schema = generate_ui_schema(cls)
            ui_schema["ui:field"] = "MoleculeField"  # This tells RJSF to use the custom component
            return ui_schema

    @classmethod
    def from_xyz(cls, content: str):
        """
        Load a molecule from an XYZ string content.

        :param content: The string content of the XYZ file.
        :return: A Molecule object.
        """
        try:
            lines = content.splitlines()
            if not lines:
                raise ValueError("Empty XYZ content")
        
            n_atoms = int(lines[0].strip())
            # Skip comment line
            elements = []
            sites = []
            for line in lines[2:2 + n_atoms]:
                data = line.strip().split()
                if len(data) >= 4:
                    elements.append(data[0])
                    sites.append([float(x) for x in data[1:4]])
            return cls.from_sites(elements, sites)
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing XYZ content: {e}")
            raise ValueError(f"Invalid XYZ format: {e}")

    

    @classmethod
    def from_cif(cls, content: str):
        """
        Load a molecule from a CIF string content.

        :param content: The string content of the CIF file.
        :return: A Molecule object.
        """
        try:
            lines = content.splitlines()
            elements = []
            sites = []
            coord_indices = [-1, -1, -1]  # indices for x, y, z coordinates
            element_index = -1
        
            # Find the loop with atomic positions
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('loop_'):
                    # Reset indices for new loop
                    coord_indices = [-1, -1, -1]
                    element_index = -1
                    header_lines = []
                
                    # Collect header lines
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith('_'):
                        header = lines[j].strip()
                        header_lines.append(header)
                        # Look for coordinate columns
                        if 'fract_x' in header or 'Cartn_x' in header:
                            coord_indices[0] = len(header_lines) - 1
                        elif 'fract_y' in header or 'Cartn_y' in header:
                            coord_indices[1] = len(header_lines) - 1
                        elif 'fract_z' in header or 'Cartn_z' in header:
                            coord_indices[2] = len(header_lines) - 1
                        elif 'type_symbol' in header or 'label' in header:
                            element_index = len(header_lines) - 1
                        j += 1
                
                    # If we found all necessary columns
                    if all(idx != -1 for idx in coord_indices) and element_index != -1:
                        # Read the data
                        while j < len(lines) and not lines[j].strip().startswith('_'):
                            data = lines[j].strip().split()
                            if len(data) >= len(header_lines):
                                elements.append(data[element_index])
                                sites.append([
                                    float(data[coord_indices[0]]),
                                    float(data[coord_indices[1]]),
                                    float(data[coord_indices[2]])
                                ])
                            j += 1
                        break

            if elements and sites:
                return cls.from_sites(elements, sites)
            else:
                raise ValueError("No atomic positions found in CIF content")
        except Exception as e:
            logger.error(f"Error parsing CIF content: {e}")
            raise ValueError(f"Invalid CIF format: {e}")

    @classmethod
    def from_sdf(cls, content: str):
        """
        Load a molecule from an SDF (molblock) string content.

        :param content: The string content of the SDF/molblock.
        :return: A Molecule object.
        """
        try:
            lines = content.splitlines()
            # Skip any leading empty lines to find the start of the molblock
            start_idx = 0
            while start_idx < len(lines) and not lines[start_idx].strip():
                start_idx += 1
            
            # Find the counts line (the one ending with V2000 or V3000)
            counts_idx = -1
            for i in range(start_idx, min(start_idx + 10, len(lines))):
                if "V2000" in lines[i] or "V3000" in lines[i]:
                    counts_idx = i
                    break
            
            if counts_idx == -1:
                # Fallback to standard line 4 if marker not found
                if len(lines) - start_idx >= 4:
                    counts_idx = start_idx + 3
                else:
                    raise ValueError("SDF content too short")

            counts_line = lines[counts_idx]
            n_atoms_str = counts_line[0:3].strip()
            if not n_atoms_str:
                raise ValueError(f"Could not find atom count in line: {counts_line!r}")
            n_atoms = int(n_atoms_str)
                
            elements = []
            sites = []
                
            # Atom block starts immediately after counts line
            for i in range(counts_idx + 1, counts_idx + 1 + n_atoms):
                line = lines[i]
                # x, y, z are in fixed widths: 10.4, 10.4, 10.4
                x = float(line[0:10].strip())
                y = float(line[10:20].strip())
                z = float(line[20:30].strip())
                # element is at 31-33
                element = line[31:34].strip()
                    
                elements.append(element)
                sites.append([x, y, z])
                    
            return cls.from_sites(elements, sites)
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing SDF content: {e}")
            raise ValueError(f"Invalid SDF format: {e}")


    @classmethod
    def from_file(cls, file_path: Path | str):
        """
        Load a molecule from a file.
    
        :param file_path: Path to the file containing the molecule data.
        :return: A Molecule object.
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if it's a file (not a directory)
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Get file extension using pathlib
        suffix = file_path.suffix.lower()
        
        if suffix == '.xyz':
            return cls.from_xyz(file_path.read_text())
        elif suffix == '.cif':
            return cls.from_cif(file_path.read_text())
        elif suffix == '.sdf' or suffix == '.mol':
            return cls.from_sdf(file_path.read_text())
        else:
            logger.error(f"Unsupported file format: {suffix}")
            raise ValueError(f"Unsupported file format: {suffix}. Only XYZ, CIF, and SDF files are supported.")

    def to_file(self, file_path: Path | str):
        """
        Save the molecule to a file.

        :param file_path: Path where the molecule data should be saved.
        """
        file_path = Path(file_path)

        # Get file extension using pathlib
        suffix = file_path.suffix.lower()

        if suffix == '.xyz':
            content = f"{len(self.atoms)}\n"
            content += f"Generated by Molecule.molecule_to_file\n"
            for atom in self.atoms:
                content += f"{atom.element} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}\n"
            file_path.write_text(content)
        else:
            logger.error(f"Unsupported file format for writing: {suffix}")
            raise ValueError(f"Unsupported file format for writing: {suffix}. Only XYZ files are currently supported.")

    def complex_hash(self, precision: int = 3, preserve_chirality: bool = False) -> str:
        """
        Generate a hash that is invariant to atom reordering, translation, and rotation,
        while preserving complete 3D geometric information including angles.

        :param precision: Number of decimal places to round values to
        :param preserve_chirality: If True, distinguishes between stereoisomers (mirror images)
        :return: A hash string that uniquely identifies the molecular structure
        """
        if not self.atoms:
            return hashlib.md5(b"empty_molecule").hexdigest()

        # Extract coordinates and elements
        coords = [[atom.x, atom.y, atom.z] for atom in self.atoms]
        elements = [atom.element for atom in self.atoms]
        n_atoms = len(coords)

        def fix_negative_zero(x):
            """Convert -0.0 to 0.0 to ensure consistent hashing."""
            rounded = round(x, precision)
            return 0.0 if rounded == 0.0 else rounded

        def distance(i, j):
            """Calculate distance between atoms i and j."""
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            return math.sqrt(dx * dx + dy * dy + dz * dz)

        def angle(i, j, k):
            """Calculate angle at atom j formed by atoms i-j-k."""
            if i == j or j == k or i == k:
                return 0.0

            # Vectors from j to i and j to k
            vec_ji = [coords[i][0] - coords[j][0], coords[i][1] - coords[j][1], coords[i][2] - coords[j][2]]
            vec_jk = [coords[k][0] - coords[j][0], coords[k][1] - coords[j][1], coords[k][2] - coords[j][2]]

            # Dot product and magnitudes
            dot_product = sum(vec_ji[l] * vec_jk[l] for l in range(3))
            mag_ji = math.sqrt(sum(vec_ji[l] * vec_ji[l] for l in range(3)))
            mag_jk = math.sqrt(sum(vec_jk[l] * vec_jk[l] for l in range(3)))

            if mag_ji < 1e-10 or mag_jk < 1e-10:
                return 0.0

            # Calculate angle (in radians)
            cos_angle = dot_product / (mag_ji * mag_jk)
            # Clamp to avoid numerical errors in acos
            cos_angle = max(-1.0, min(1.0, cos_angle))
            return math.acos(cos_angle)

        def triangle_area(i, j, k):
            """Calculate area of triangle formed by three atoms (invariant geometric property)."""
            if len(set([i, j, k])) != 3:
                return 0.0

            # Side lengths
            a = distance(j, k)
            b = distance(i, k)
            c = distance(i, j)

            # Heron's formula
            s = (a + b + c) / 2
            if s <= a or s <= b or s <= c:
                return 0.0

            area_squared = s * (s - a) * (s - b) * (s - c)
            return math.sqrt(max(0, area_squared)) if area_squared > 0 else 0.0

        def tetrahedron_volume(i, j, k, l, preserve_chirality=False):
            """Calculate volume of tetrahedron formed by four atoms."""
            if len(set([i, j, k, l])) != 4:
                return 0.0

            # Vectors from i to other points
            v1 = [coords[j][m] - coords[i][m] for m in range(3)]
            v2 = [coords[k][m] - coords[i][m] for m in range(3)]
            v3 = [coords[l][m] - coords[i][m] for m in range(3)]

            # Scalar triple product (determinant)
            det = (v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
                   v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
                   v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]))

            if preserve_chirality:
                # Preserve sign for chirality detection
                return det / 6.0
            else:
                # Use absolute value for reflection-invariant comparison
                return abs(det) / 6.0

        # Create comprehensive geometric descriptors for each atom
        atom_descriptors = []
        for i, element in enumerate(elements):
            # 1. Distance-based environment (as before)
            distances = []
            for j in range(n_atoms):
                if i != j:
                    dist = fix_negative_zero(distance(i, j))
                    distances.append((dist, elements[j]))
            distances.sort()

            # 2. Angular environment - angles centered at this atom
            angles = []
            for j in range(n_atoms):
                for k in range(j + 1, n_atoms):
                    if i != j and i != k:
                        ang = fix_negative_zero(angle(j, i, k))
                        # Sort element names to ensure invariance to ordering
                        elem_pair = tuple(sorted([elements[j], elements[k]]))
                        angles.append((ang, elem_pair))
            angles.sort()

            # 3. Triangle areas involving this atom (geometric invariant)
            triangles = []
            for j in range(n_atoms):
                for k in range(j + 1, n_atoms):
                    if i != j and i != k and j != k:
                        area = fix_negative_zero(triangle_area(i, j, k))
                        # Sort element names to ensure invariance to ordering
                        elem_triple = tuple(sorted([elements[j], elements[k]]))
                        triangles.append((area, elem_triple))
            triangles.sort()

            # 4. Tetrahedron volumes involving this atom (chirality-sensitive)
            tetrahedra = []
            for j in range(n_atoms):
                for k in range(j + 1, n_atoms):
                    for l in range(k + 1, n_atoms):
                        if len(set([i, j, k, l])) == 4:
                            vol = fix_negative_zero(tetrahedron_volume(i, j, k, l, preserve_chirality))
                            # Sort element names to ensure invariance to ordering
                            elem_quad = tuple(sorted([elements[j], elements[k], elements[l]]))
                            tetrahedra.append((vol, elem_quad))
            tetrahedra.sort()

            # 5. Create comprehensive descriptor
            descriptor = (
                element,
                tuple(distances),
                tuple(angles[:8]),  # Limit to most significant angles
                tuple(triangles[:6]),  # Limit to most significant triangles
                tuple(tetrahedra[:4])  # Limit to most significant tetrahedra
            )
            atom_descriptors.append(descriptor)

        # Sort atom descriptors for canonical ordering (atom reordering invariance)
        atom_descriptors.sort()

        # Create hash from the comprehensive geometric representation
        hash_input = str(atom_descriptors).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()



NEW_MOLECULE_LIST = True
#
# if not NEW_MOLECULE_LIST:
#     @simstack_model
#     class MoleculeList(Model):
#         """
#         A class representing a list of Molecule objects.
#
#         Attributes:
#             molecules (List[Molecule]): A list of Molecule objects.
#         """
#         field_name: str = "MoleculeList"
#         molecules: List[Molecule] = Field(default_factory=list)
#
#         @model_validator(mode="before")
#         @classmethod
#         def ensure_fieldname(cls, data):
#             """Ensure fieldname is set for existing documents"""
#             if isinstance(data, dict) and "field_name" not in data:
#                 data["field_name"] = cls.__name__
#             return data
#
#         def append(self, molecule: Molecule):
#             return self.add_molecule(molecule)
#
#         def add_molecule(self, molecule: Molecule):
#             """
#             Add a molecule to the list.
#
#             :param molecule: A Molecule object to be added to the list.
#             """
#             self.molecules.append(molecule)
#
# else:
@simstack_model
class MoleculeList(Model, ObjectListMixin[Molecule]):
    """
    A class representing a list of Molecule objects using references.

    Attributes:
        elements (List[ObjectId]): A list of Molecule references.
    """
    field_name: str = "MoleculeList"
    elements: List[ObjectId] = Field(default_factory=list)

    def __init__(self, **data):
        data, cache = self._normalize_elements_for_init(data)
        Model.__init__(self, **data)
        if cache is not None:
            self._set_cache(cache)

    def __iter__(self) -> Iterator[Molecule]:
        return ObjectListMixin.__iter__(self)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    def add_molecule(self, molecule: Molecule):
        self.append(molecule)
    # def append(self, molecule: Molecule):
    #     return self.add_molecule(molecule)
    #
    # def add_molecule(self, molecule: Molecule):
    #     """
    #     Add a molecule to the list.
    #
    #     :param molecule: A Molecule object to be added to the list.
    #     """
    #     if not hasattr(self, "_molecule_cache"):
    #         self._molecule_cache = []
    #     self._molecule_cache.append(molecule)
    #     self.elements.append(molecule.id)

    # @property
    # def molecules(self) -> List[Molecule]:
    #     cache = self._g
    #     return self.elements
    #
    # async def get_all_molecules(self) -> List[Molecule]:
    #     """Load all molecules from the database and cache them."""
    #     if not hasattr(self, "_molecule_cache") or len(self._molecule_cache) != len(self.elements):
    #         self._molecule_cache = [mol async for mol in self]
    #     return self._molecule_cache
    #
    # async def __aiter__(self):
    #     """Async iterator that yields Molecule objects from the database.
    #
    #     This allows using 'async for mol in molecule_list' syntax when the
    #     MoleculeList stores references (ObjectIds) instead of embedded molecules.
    #     """
    #     from simstack.core.context import context
    #
    #     # Ensure context is initialized
    #     if not context.initialized:
    #         await context.initialize()
    #
    #     db = context.db
    #
    #     for element_ref in self.elements:
    #         mol = await db.find_one(Molecule, Molecule.id == element_ref)
    #         if mol is None:
    #             logger.warning(f"Could not find Molecule with id {element_ref} in database")
    #             continue
    #         yield mol

    @classmethod
    def from_file(cls, file_path: Path | str, start: int = 0, number: Optional[int] = None) -> "MoleculeList":
        """
        Load multiple molecules from a file.

        :param file_path: Path to the file containing molecule data (supports .xyz, .cif, .sdf formats with multiple molecules).
        :param start: Index of the first molecule to read (0-based indexing).
        :param number: Maximum number of molecules to read. If None, read all molecules from start.
        :return: A MoleculeList object containing all molecules from the file.
        """

        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if it's a file (not a directory)
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        suffix = file_path.suffix.lower()
        content = file_path.read_text()
        molecule_list = cls()
        molecule_index = 0

        if suffix == '.xyz':
            # Parse multiple XYZ molecules separated by blank lines or consecutive molecule blocks
            lines = content.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line and line[0].isdigit():
                    try:
                        n_atoms = int(line)
                        # Extract this molecule's block
                        molecule_block = '\n'.join(lines[i:i + n_atoms + 2])

                        # Check if we should include this molecule
                        if molecule_index >= start:
                            current_len = len(molecule_list.elements) if NEW_MOLECULE_LIST else len(molecule_list.molecules)
                            if number is None or current_len < number:
                                molecule = Molecule.from_xyz(molecule_block)
                                molecule_list.add_molecule(molecule)

                        molecule_index += 1
                        i += n_atoms + 2

                        # Stop if we've read the requested number of molecules
                        current_len = len(molecule_list.elements) if NEW_MOLECULE_LIST else len(molecule_list.molecules)
                        if number is not None and current_len >= number:
                            break
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid XYZ block at line {i}: {e}")
                        i += 1
                else:
                    i += 1

        elif suffix == '.sdf' or suffix == '.mol':
            # Parse multiple SDF molecules separated by $$$$
            molecule_blocks = content.split('$$$$')
            for block in molecule_blocks:
                block = block.strip()
                if block:
                    try:
                        # Check if we should include this molecule
                        if molecule_index >= start:
                            current_len = len(molecule_list.elements) if NEW_MOLECULE_LIST else len(molecule_list.molecules)
                            if number is None or current_len < number:
                                molecule = Molecule.from_sdf(block)
                                molecule_list.add_molecule(molecule)
                        # Stop if we've read the requested number of molecules
                        current_len = len(molecule_list.elements) if NEW_MOLECULE_LIST else len(molecule_list.molecules)
                        if number is not None and current_len >= number:
                            break
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid SDF block: {e}")
                        molecule_index += 1
        elif suffix == '.cif':
            # CIF files typically contain one structure, but we support the format
            try:
                # Check if we should include this molecule
                current_len = len(molecule_list.elements) if NEW_MOLECULE_LIST else len(molecule_list.molecules)
                if molecule_index >= start and (number is None or current_len < number):
                    molecule = Molecule.from_cif(content)
                    molecule_list.add_molecule(molecule)
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing CIF file: {e}")
                raise ValueError(f"Invalid CIF format: {e}")
        else:
            logger.error(f"Unsupported file format: {suffix}")
            raise ValueError(f"Unsupported file format: {suffix}. Only XYZ, CIF, and SDF files are supported.")

        return molecule_list

    # on 26.07. Decision by Prof. W - Async should no longer be used here in molecules.py
        #async def to_file(self, file_path: Path | str):
    def to_file(self, file_path: Path | str):
        """
        Save the molecule list to a file.

        :param file_path: Path where the molecule data should be saved.
        """
        file_path = Path(file_path)

        # Get file extension using pathlib
        suffix = file_path.suffix.lower()

        molecules = []
        if NEW_MOLECULE_LIST:
            #async for mol in self:
                #why not do this as a comprehension?- is there anything to watch out for with the list class?
                #otherwise this would be more efficient as molecules= [mol for mol in self]
                for mol in self:
                    molecules.append(mol)
        else:
            molecules = self.molecules

        if suffix == '.xyz':
            content = ""
            for index, molecule in enumerate(molecules):
                content += f"{len(molecule.atoms)}\n"
                content += f"Frame {index:04d}\n"
                for atom in molecule.atoms:
                    content += f"{atom.element:3s} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}\n"
            file_path.write_text(content)
        elif suffix == '.sdf' or suffix == '.mol':
            content = ""
            for molecule in molecules:
                # Write molecule header (3 lines: name, program, comment)
                content += "Generated by MoleculeList.to_file\n"
                content += "\n"
                content += "\n"

                # Counts line
                n_atoms = len(molecule.atoms)
                content += f"{n_atoms:3d}  0  0  0  0  0  0  0  0  0999 V2000\n"

                # Atom block
                for atom in molecule.atoms:
                    content += f"{atom.x:10.4f}{atom.y:10.4f}{atom.z:10.4f} {atom.element:3s} 0  0  0  0  0  0  0  0  0  0  0  0\n"

                # End of molecule marker
                content += "M  END\n"
                content += "$$$$\n"
            file_path.write_text(content)
        else:
            logger.error(f"Unsupported file format for writing: {suffix}")
            raise ValueError(
                f"Unsupported file format for writing: {suffix}. Only XYZ and SDF files are currently supported.")

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["ui:field"] = "MoleculeListField"  # This tells RJSF to use the custom component
        return ui_schema
