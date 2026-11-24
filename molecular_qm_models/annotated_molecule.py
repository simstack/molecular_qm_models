import math
from typing import List

from odmantic import EmbeddedModel, Field, Model, Reference
from pydantic import model_validator

from molecular_qm_models.molecule import Molecule
from simstack.models import simstack_model
from simstack.util.generate_ui_schema import generate_ui_schema


@simstack_model
class SphereAnnotation(EmbeddedModel):
    """A colored sphere drawn at one coordinate on an annotated molecule."""

    field_name: str = "SphereAnnotation"
    coordinate: List[float]
    radius: float
    color: str
    transparency: float

    @model_validator(mode="after")
    def validate_sphere(self):
        if len(self.coordinate) != 3 or any(not math.isfinite(value) for value in self.coordinate):
            raise ValueError("coordinate must be three finite numbers")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be a positive finite number")
        if not self.color.strip():
            raise ValueError("color is required")
        if not math.isfinite(self.transparency) or self.transparency < 0 or self.transparency > 1:
            raise ValueError("transparency must be a finite number between 0 and 1")
        return self


@simstack_model
class BondAnnotation(EmbeddedModel):
    """A colored line drawn between two coordinates on an annotated molecule."""

    field_name: str = "BondAnnotation"
    points: List[List[float]]
    color: str
    thickness: float
    transparency: float

    @model_validator(mode="after")
    def validate_bond(self):
        if len(self.points) != 2:
            raise ValueError("points must contain exactly two coordinates")
        for point in self.points:
            if len(point) != 3 or any(not math.isfinite(value) for value in point):
                raise ValueError("each point must be three finite numbers")
        if not self.color.strip():
            raise ValueError("color is required")
        if not math.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("thickness must be a positive finite number")
        if not math.isfinite(self.transparency) or self.transparency < 0 or self.transparency > 1:
            raise ValueError("transparency must be a finite number between 0 and 1")
        return self


@simstack_model
class AnnotatedMolecule(Model):
    """A molecule reference plus the spheres and bonds drawn on top of it."""

    field_name: str = "AnnotatedMolecule"
    molecule: Molecule = Reference()
    sphere_annotations: List[SphereAnnotation] = Field(default_factory=list)
    bond_annotations: List[BondAnnotation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["ui:field"] = "AnnotatedMoleculeField"
        return ui_schema
