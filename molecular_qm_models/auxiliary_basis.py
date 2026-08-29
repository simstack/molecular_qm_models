from enum import Enum
from odmantic import Model, Field, EmbeddedModel
from pydantic import model_validator
from simstack.models import simstack_model
from simstack.util.generate_ui_schema import generate_ui_schema
from typing import Dict, Any, List


class AuxBasisEnum(str, Enum):
    """
    Enum for auxiliary basis sets used in density fitting calculations.
    """
    NONE = "none"   # embedded models cannot be Optional so we need a default that is none
    Def2_J = "def2/J"
    Def2_JK = "def2/JK"
    Def2_TZVP_C = "def2-TZVP/C"
    Def2_SVP_C = "def2-SVP/C"
    AutoAux = "AutoAux"
    CC_PVDZ_J = "cc-pVDZ/J"
    CC_PVTZ_J = "cc-pVTZ/J"
    CC_PVQZ_J = "cc-pVQZ/J"
    CC_PVDZ_JK = "cc-pVDZ/JK"
    CC_PVTZ_JK = "cc-pVTZ/JK"
    CC_PVQZ_JK = "cc-pVQZ/JK"
    CC_PVDZ_C = "cc-pVDZ/C"
    CC_PVTZ_C = "cc-pVTZ/C"
    CC_PVQZ_C = "cc-pVQZ/C"
    CC_PWCVDZ_J = "cc-pwCVDZ/J"
    CC_PWCVTZ_J = "cc-pwCVTZ/J"
    SVP_J = "SVP/J"
    TZVP_J = "TZVP/J"
    SARC_J = "SARC/J"
    SARC2_J = "SARC2/J"
    Universal_J = "Universal/J"


@simstack_model
class AuxBasis(EmbeddedModel):
    """
    A class representing an auxiliary basis set for density fitting in quantum mechanical calculations.
    """
    field_name: str = "AuxBasis"

    aux_basis: AuxBasisEnum = Field(AuxBasisEnum.NONE,
                               json_schema_extra={
                                   "enum": [e.value for e in AuxBasisEnum],
                               }
                              )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    # @property
    # def aux_basis(self) -> str:
    #     return self.aux_basis_str
    #
    # @aux_basis.setter
    # def aux_basis(self, value: str):
    #     if value not in [e.value for e in AuxBasisEnum]:
    #         raise ValueError(f"Invalid auxiliary basis set: {value}")
    #     self.aux_basis_str = value
    #
    @classmethod
    def cleaned_json(cls, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify the JSON schema to add a dependency between basis_set_str and aux_basis_str
        such that only compatible auxiliary basis sets are shown based on the main basis set.
        
        Args:
            json_schema: The original JSON schema for the AuxBasis class
            
        Returns:
            Modified JSON schema with dependencies
        """
        # Define the compatible auxiliary basis sets for each main basis family
        basis_aux_mapping = {
            # For def2 family basis sets
            "Def2-SVP": ["def2/J", "def2/JK", "AutoAux"],
            "Def2-TZVP": ["def2/J", "def2/JK", "def2-TZVP/C", "AutoAux"],
            "Def2-QZVP": ["def2/J", "def2/JK", "AutoAux"],
            
            # For cc-pVXZ family
            "cc-pVDZ": ["cc-pVDZ/J", "cc-pVDZ/JK", "cc-pVDZ/C", "AutoAux"],
            "cc-pVTZ": ["cc-pVTZ/J", "cc-pVTZ/JK", "cc-pVTZ/C", "AutoAux"],
            
            # For other basis sets
            "STO3G": ["AutoAux"],
            "STO6G": ["AutoAux"],
            "6-31g": ["AutoAux"],
            "6-31g*": ["AutoAux"],
            "6-31g**": ["AutoAux"],
        }
        
        # Create the anyOf construct for the schema
        any_of_conditions = []
        for basis, aux_bases in basis_aux_mapping.items():
            any_of_conditions.append({
                "properties": {
                    "basis_set": {"enum": [basis]}
                },
                "anyOf": [
                    {
                        "properties": {
                            "aux_basis": {"enum": aux_bases}
                        }
                    }
                ]
            })
        
        # Add a default case for any basis not explicitly listed
        default_aux_bases = ["AutoAux", "Universal/J"]
        any_of_conditions.append({
            "not": {
                "properties": {
                    "basis_set": {"enum": list(basis_aux_mapping.keys())}
                }
            },
            "properties": {
                "aux_basis": {"enum": default_aux_bases}
            }
        })
        
        # Add the anyOf to the schema
        json_schema["anyOf"] = any_of_conditions
        
        return json_schema
