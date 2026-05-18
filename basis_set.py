from enum import Enum
from typing import List, Dict, Any

from odmantic import Field, EmbeddedModel, Model
from pydantic import model_validator

from .auxiliary_basis import AuxBasis, AuxBasisEnum
from simstack.models import simstack_model


class BasisSetEnum(str, Enum):
    """
    Enum for basis sets used in quantum chemistry calculations.
    """
    STO3G = "STO3G"
    STO6G = "STO6G"
    B631G = "6-31G"
    B631G_star = "6-31G*"
    B631G_star_star = "6-31G**"
    cc_pVDZ = "cc-pVDZ"
    cc_pVTZ = "cc-pVTZ"
    cc_pVQZ = "cc-pVQZ"
    cc_pV5Z = "cc-pV5Z"
    Def2_SVP = "def2-SVP"
    Def2_TZVP = "def2-TZVP"
    Def2_QZVP = "def2-QZVP"
    aug_cc_pVDZ = "aug-cc-pVDZ"
    aug_cc_pVTZ = "aug-cc-pVTZ"

@simstack_model
class BasisSet(EmbeddedModel):
    """
    A class representing a basis set for quantum mechanical calculations.
    """
    field_name: str = "BasisSet"
    basis_set: BasisSetEnum = Field(BasisSetEnum.Def2_SVP,
                               # this is absolutely required to have a drop-down instead
                               # of a string field for the enum
                               json_schema_extra={
                                   "enum": [e.value for e in BasisSetEnum],

                               })

    aux_basis: AuxBasis = Field(default=AuxBasis(aux_basis=AuxBasisEnum.NONE))

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    def make_table_entries(self,**kwargs):
        return {'basis_set': str(self.basis_set.value) , 'aux_basis': str(self.aux_basis.aux_basis.value)}

    def make_column_defs_instance(self,**kwargs) -> List[Dict[str, Any]]:
        field_prefix = kwargs.get("field_prefix", "")
        return [{'headerName': 'Basis Set', 'field': field_prefix + 'basis_set' },
                {'headerName': 'Aux Basis', 'field': field_prefix + 'aux_basis' }, ]

@simstack_model
class BasisSetModel(Model):
    field_name: str = "BasisSetModel"
    basis_set: BasisSet = Field(default=BasisSet(basis_set=BasisSetEnum.Def2_SVP,aux_basis=AuxBasis(aux_basis=AuxBasisEnum.Def2_J)))

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

