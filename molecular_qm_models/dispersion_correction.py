from enum import Enum
from odmantic import Model, Field, EmbeddedModel
from pydantic import model_validator
from typing import Optional, Dict, Any
from simstack.core.context import context
from simstack.models import simstack_model
from simstack.util.ui_tools import ui_make_properties_optional

# NOTE: it is essential to derive from both str and Enum !
class DispersionCorrectionEnum(str,Enum):
    """
    Enumeration of available dispersion correction versions.
    """
    NONE = "NONE"
    D2 = "D2"
    D3 = "D3"
    D3BJ = "D3BJ"  # D3 with Becke-Johnson damping
    D4 = "D4"
    NL = "NL"  # Non-local van der Waals functional

    @classmethod
    def get_description(cls, version):
        """
        Returns a description of the dispersion correction version.
        """
        descriptions = {
            cls.NONE: "No dispersion correction",
            cls.D2: "Original DFT-D2 correction by Grimme",
            cls.D3: "DFT-D3 correction with zero damping",
            cls.D3BJ: "DFT-D3 correction with Becke-Johnson damping",
            cls.D4: "DFT-D4 correction with advanced charge-dependent coefficients",
            cls.NL: "Non-local van der Waals functional (more accurate but computationally intensive)"
        }
        return descriptions.get(version, "No description available")


@simstack_model
class DispersionCorrection(EmbeddedModel):
    """
    Dispersion corrections.

    Attributes:
        value: The type of dispersion correction from DispersionCorrectionEnum

    """
    field_name: str = "DispersionCorrection"
    value: DispersionCorrectionEnum = Field(default=DispersionCorrectionEnum.NONE,
        json_schema_extra={
            "enum": [e.value for e in DispersionCorrectionEnum],
            "description": "Version of the dispersion correction to use"
        }
    )
    #
    # include_three_body: bool = Field(
    #     default=True,
    #     json_schema_extra={
    #         "description": "Whether to include three-body dispersion terms (ATM term)"
    #     }
    # )
    #
    # cutoff_radius: float = Field(
    #     default=1.0,
    #     json_schema_extra={
    #         "description": "Cutoff radius for dispersion interactions in Angstroms",
    #         "minimum": 0
    #     }
    # )
    #
    # scale_factors: Dict[str, float] = Field(
    #     default_factory=lambda: {
    #         "s6": 1.0,  # Scale factor for 6th-order term
    #         "s8": 1.0,  # Scale factor for 8th-order term
    #         "sr6": 1.0,  # Short-range cutoff for 6th-order term
    #         "sr8": 1.0,  # Short-range cutoff for 8th-order term
    #         "alpha6": 14.0  # Parameter for D3BJ damping
    #     },
    #     json_schema_extra={
    #         "description": "Scale factors for different terms in the dispersion correction"
    #     }
    # )
    #
    # custom_parameters: Dict[str, Any] = Field(
    #     default_factory=dict,
    #     json_schema_extra={
    #         "description": "Custom parameters for the dispersion correction"
    #     }
    # )

    # @property
    # def version(self) -> DispersionCorrectionEnum:
    #     """
    #     Returns the version of the dispersion correction as an enum.
    #     """
    #     return DispersionCorrectionEnum(self.version_str)
    #
    # @version.setter
    # def version(self, value: DispersionCorrectionEnum):
    #     """
    #     Sets the version of the dispersion correction.
    #     """
    #     if isinstance(value, DispersionCorrectionEnum):
    #         self.version_str = value.value
    #     else:
    #         raise ValueError("Invalid version type. Must be DispersionVersionEnum.")

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def json_schema(cls):
        """
        Generate a JSON schema for this model.

        :return: A JSON schema dictionary.
        """
        # For this class, we don't use recursive_json_schema directly
        # Instead, we start with the basic model schema and customize it
        schema = cls.model_json_schema()
        schema['title'] = cls.__name__
        schema['description'] = "Parameters for dispersion corrections"

        # Define scale_factors schema in $defs
        if '$defs' not in schema:
            schema['$defs'] = {}

        # Add DispersionVersionEnum schema definition
        schema['$defs']['DispersionVersionEnum'] = {
            "type": "string",
            "enum": [e.value for e in DispersionCorrectionEnum],
            "title": "Dispersion Version",
            "description": "Version of the dispersion correction "
        }

        # schema['$defs']['scale_factors'] = {
        #     "type": "object",
        #     "properties": {
        #         "s6": {"type": "number", "title": "S6", "default": 1.0},
        #         "s8": {"type": "number", "title": "S8", "default": 1.0},
        #         "sr6": {"type": "number", "title": "SR6", "default": 1.0},
        #         "sr8": {"type": "number", "title": "SR8", "default": 1.0},
        #         "alpha6": {"type": "number", "title": "Alpha6", "default": 14.0}
        #     },
        #     "title": "Scale Factors"
        # }
        # schema["properties"]["scale_factors"] = {
        #     "$ref": "#/$defs/scale_factors"
        # }
        # sub_schema = ui_make_properties_optional(schema,
        #                                          ["value", "include_three_body", "cutoff_radius", "scale_factors",
        #                                           "custom_parameters"], "use_dispersion_correction")
        #
        # ui_make_properties_optional(sub_schema, ["include_three_body","cutoff_radius","scale_factors","custom_parameters"], "Custom Parameters")

        return schema

    @classmethod
    def ui_base_schema(cls):
        return {
            "id": {
                "ui:widget": "hidden"
            },
            "ui:options": {
                "ui:foldable" : True
            },
            "*": {
                "ui:order": [
                    "use_dispersion_correction",
                ],
                "use_dispersion_correction": {
                    "ui:order": [
                        "value",
                        "include_three_body",
                        "cutoff_radius",
                        "scale_factors",
                        "custom_parameters"
                    ]
                }
            },
            "scale_factors": {
                "ui:order": [
                    "s6",
                    "s8",
                    "sr6",
                    "sr8",
                    "alpha6"
                ]
            }
        }

    def get_version_description(self) -> str:
        """
        Returns a description of the current dispersion correction version.
        """
        return DispersionCorrectionEnum.get_description(self.value)

    def get_parameter_string(self) -> str:
        """
        Returns a formatted string of the current parameters for input files.
        """
        params = []
        if self.value == DispersionCorrectionEnum.D3BJ:
            params.append("D3BJ")
        elif self.value == DispersionCorrectionEnum.D3:
            params.append("D3")
        elif self.value == DispersionCorrectionEnum.D4:
            params.append("D4")

        if not self.include_three_body:
            params.append("noabc")

        if self.cutoff_radius is not None:
            params.append(f"cutoff={self.cutoff_radius}")

        return " ".join(params)

    def is_compatible_with_functional(self, functional_name: str) -> bool:
        """
        Checks if the dispersion correction is compatible with a given functional.

        Args:
            functional_name: Name of the density functional

        Returns:
            bool: True if compatible, False otherwise
        """
        # This could be expanded with a more comprehensive compatibility check
        incompatible = {
            DispersionCorrectionEnum.D2: ["B97D", "WB97X-D"],  # Functionals that already include D2
            DispersionCorrectionEnum.D3: ["WB97M-D3", "B97M-D3"],  # Functionals that already include D3
            DispersionCorrectionEnum.D4: [""],  # Functionals incompatible with D4
        }
        return functional_name not in incompatible.get(self.version, [])

    def __str__(self) -> str:
        """
        Returns a string representation of the dispersion correction.
        """
        if self.value != DispersionCorrectionEnum.NONE:
            return f"DFT-{self.value.value}"
        else:
            return ""


@simstack_model
class ModelWithDispersion(Model):
    field_name: str = "ModelWithDispersion"
    dispersion: DispersionCorrection

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data


async def main():
    context.initialize()
    # Example usage
    dispersion = DispersionCorrection(
        use_dispersion_correction=True,
        value=DispersionCorrectionEnum.D3,
        include_three_body=True,
        cutoff_radius=1.2,
        scale_factors={"s6": 1.0, "s8": 1.0, "sr6": 1.0, "sr8": 1.0, "alpha6": 14.0},
        custom_parameters={"custom_param": "value"}
    )
    print(dispersion)
    print(dispersion.get_version_description())
    print(dispersion.get_parameter_string())

    model = ModelWithDispersion(dispersion=dispersion)
    await context.db.engine.save(model)

    dict_repr = dispersion.model_dump()

    print(dict_repr)
    # Create a new instance from the dictionary representation
    new_dispersion = DispersionCorrection.model_validate(dict_repr)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
