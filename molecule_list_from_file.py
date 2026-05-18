import asyncio
import gzip
from pathlib import Path

from molecular_qm_models import compute_smiles, compute_iupac_name
from simstack.core.context import context
from simstack.core.node import node
from simstack.models.files import FileStack
from molecular_qm_models import Molecule, MoleculeList
from applications.electronic_structure import iter_multixyz_frames, iter_sdf_frames


@node
async def molecule_list_from_file(file: FileStack, **kwargs):
    local_copy = file.get()

    # Check if the file is gzip-compressed
    file_path = Path(local_copy)
    if file_path.suffix == '.gz':
        # Decompress the file
        decompressed_path = file_path.with_suffix('')
        with gzip.open(local_copy, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        local_copy = Path(decompressed_path)
    
    local_copy = Path(local_copy)
    suffix = local_copy.suffix.lower()
    content = local_copy.read_text()
    
    molecule_list = MoleculeList()
    
    if suffix == '.xyz':
        for frame in iter_multixyz_frames(content):
            mol = Molecule.from_xyz(frame.to_xyz_string())
            await context.db.save(mol)
            molecule_list.add_molecule(mol)
    elif suffix == '.sdf':
        for block in iter_sdf_frames(content):
            mol = Molecule.from_sdf(block)

            molecule_list.add_molecule(mol)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Only .sdf and .xyz are supported.")

    for molecule in molecule_list.molecules:
        molecule.smiles = compute_smiles(molecule)
        molecule.formula = compute_iupac_name(molecule)
        await context.db.save(molecule)
    return molecule_list

async def main():
    await context.initialize()
    path = Path(__file__).parent / "data" / "twowater.xyz"
    file = FileStack.from_local_file(path,in_memory=True,is_hashable=True)
    molecule_list = await molecule_list_from_file(file)
    print(molecule_list)

if __name__ == "__main__":
    asyncio.run(main())
