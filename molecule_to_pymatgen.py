from pymatgen.core import Molecule as PymatgenMolecule
from typing import Optional
import logging

from molecular_qm_models import Molecule

logger = logging.getLogger(__name__)

def molecule_to_pymatgen(molecule: Molecule,
                         charge: Optional[int] = None, 
                         spin_multiplicity: Optional[int] = None) -> PymatgenMolecule:
    """
    Convert a Molecule object from the custom class to a Pymatgen Molecule object.
    
    Args:
        molecule: The custom Molecule object to convert
        charge: Optional charge to override the molecule's charge property
        spin_multiplicity: Optional spin multiplicity to override the molecule's spin_multiplicity property
        
    Returns:
        A Pymatgen Molecule object

    """
    try:
        # Extract species and coordinates from atoms
        species = []
        coords = []
        
        for atom in molecule.atoms:
            species.append(atom.element)
            coords.append([atom.x, atom.y, atom.z])
        
        # Use provided charge and spin_multiplicity if given, otherwise use molecule properties
        final_charge = charge if charge is not None else molecule.charge
        final_spin = spin_multiplicity if spin_multiplicity is not None else molecule.spin_multiplicity
        
        # Create and return the Pymatgen Molecule
        return PymatgenMolecule(
            species=species,
            coords=coords,
            charge=final_charge,
            spin_multiplicity=final_spin
        )
    except Exception as e:
        logger.error(f"Error converting molecule to Pymatgen format: {str(e)}")
        raise ValueError(f"Failed to convert molecule to Pymatgen format: {str(e)}")



def pymatgen_to_molecule(pymatgen_mol: PymatgenMolecule) -> Molecule:
    """
    Convert a Pymatgen Molecule object to a custom Molecule object.

    Args:
        pymatgen_mol: The Pymatgen Molecule object to convert

    Returns:
        A custom Molecule object

    Example:
        >>> from pymatgen.core import Molecule as PymatgenMolecule
        >>> from simstack.electronic_structure.electronic_structure.molecule_conversion_to_pymatgen import pymatgen_to_molecule
        >>> # Create a Pymatgen water molecule
        >>> pymt_water = PymatgenMolecule(
        ...     species=["O", "H", "H"],
        ...     coords=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],
        ...     charge=0,
        ...     spin_multiplicity=1
        ... )
        >>> # Convert to custom Molecule
        >>> water = pymatgen_to_molecule(pymt_water)
    """
    try:
        # Import here to avoid circular imports
        from molecular_qm_models import Molecule, Atom

        # Create a new custom Molecule object
        molecule = Molecule()

        # Set properties from Pymatgen Molecule
        molecule.properties['charge'] = pymatgen_mol.charge
        molecule.properties['spin_multiplicity'] = pymatgen_mol.spin_multiplicity

        # Add atoms from Pymatgen Molecule
        for i, site in enumerate(pymatgen_mol.sites):
            element = str(site.specie.symbol)
            x, y, z = site.coords

            # Create and add Atom to the Molecule
            atom = Atom(
                element=element,
                x=float(x),
                y=float(y),
                z=float(z)
            )
            molecule.add_atom(atom)

        return molecule
    except Exception as e:
        logger.error(f"Error converting Pymatgen molecule to custom format: {str(e)}")
        raise ValueError(f"Failed to convert Pymatgen molecule to custom format: {str(e)}")