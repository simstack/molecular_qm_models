import asyncio
from datetime import datetime
from pathlib import Path
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import DataSetMetadata, DataSetTupleSection, DataSetTuple, StringData
from simstack.models.files import FileStack

from molecular_qm_models import MoleculeList
from .molecule_list_from_file import molecule_list_from_file


@node
async def database_from_molecules(molecule_list: MoleculeList, name: StringData, **kwargs):

    metadata = DataSetMetadata(field_name="molecule-dataset",data = {
        "name": name.value,
        "created_at": datetime.now()
    })
    database = DataSetTuple(field_name="molecules", metadata=metadata)
    section = DataSetTupleSection()
    database["molecules"] = section

    for molecule in molecule_list.molecules:
        smiles_data = StringData(field_name="ext-smiles",value=molecule.smiles)
        await context.db.save(smiles_data)
        formula_data = StringData(field_name="ext-formula",value=molecule.formula)
        await context.db.save(formula_data)
        section.append((smiles_data,formula_data,molecule))

    return database


async def main():
    await context.initialize()
    path = Path(__file__).parent / "data" / "twowater.xyz"
    file = FileStack.from_local_file(path,in_memory=True,is_hashable=True)
    molecule_list = await molecule_list_from_file(file)
    print(molecule_list)
    name = StringData(field_name="db_name", value="test-data")
    database = await database_from_molecules(molecule_list, name)


if __name__ == "__main__":
    asyncio.run(main())
