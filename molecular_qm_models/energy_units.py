from enum import Enum

from odmantic import EmbeddedModel
from simstack.models import simstack_model

class MolecularEnergyUnitEnum(str,Enum):
    HARTREE = "hartree"
    EV = "ev"
    KJ_PER_MOL = "kj/mol"
    KCAL_PER_MOL = "kcal/mol"
    
@simstack_model
class MolecularEnergyUnit(EmbeddedModel):
    unit: MolecularEnergyUnitEnum
    
def convert_energy_unit(source_unit: MolecularEnergyUnitEnum, value: float, target_unit: MolecularEnergyUnitEnum) -> float:
    if source_unit == target_unit:
        return value

    # Factors relative to Hartree
    # Using CODATA 2018 values and exact definitions
    # 1 Hartree = 27.211386245988 eV
    # 1 Hartree = 2625.4996394799 kJ/mol
    # 1 kcal = 4.184 kJ (exactly)
    # 1 Hartree = 2625.4996394799 / 4.184 kcal/mol = 627.50947406307 kcal/mol
    
    factors = {
        MolecularEnergyUnitEnum.HARTREE: 1.0,
        MolecularEnergyUnitEnum.EV: 27.211386245988,
        MolecularEnergyUnitEnum.KJ_PER_MOL: 2625.4996394799,
        MolecularEnergyUnitEnum.KCAL_PER_MOL: 627.50947406307,
    }

    # Convert to Hartree first: value / factors[source_unit]
    # Then convert to target: (value / factors[source_unit]) * factors[target_unit]
    return value * (factors[target_unit] / factors[source_unit])
    
    
