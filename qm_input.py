from enum import Enum
from typing import Optional, List, Self
from odmantic import Model, Field, Reference, EmbeddedModel
from pydantic import model_validator

from .basis_set import BasisSet
from .density_functional import Functional
from .molecule import Molecule

from simstack.models import simstack_model
from simstack.models.file_list import FileList
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


class QMMethod(str, Enum):
    """Enum for single reference quantum chemistry calculation methods.

    This mirrors and extends the legacy ``QMMethod`` definition from the
    old QMInput model, including the DLPNO variants that are still used by
    the ORCA helpers (e.g. ``set_method_and_basis_set_for_non_casscf_methods``).
    """

    HF = "HF"
    DFT = "DFT"
    MP2 = "MP2"
    CCSD = "CCSD"
    CCSD_T = "CCSD(T)"
    DLPNO_CCSD = "DLPNO-CCSD"
    DLPNO_CCSD_T = "DLPNO-CCSD(T)"
    CIS = "CIS"
    TDDFT = "TDDFT"
    RPA = "RPA"
    DFTMRCI = "DFTMRCI"
    CASSCF = "CASSCF"


class SCFAccuracy(str, Enum):
    """Enum for SCF convergence accuracy levels"""
    Sloppy = "Sloppy"  # very weak convergence
    Loose = "Loose"  # still weak convergence
    Medium = "Medium"  # intermediate accuracy
    Strong = "Strong"  # stronger
    Tight = "Tight"  # still stronger
    VeryTight = "VeryTight"  # even stronger
    Extreme = "Extreme"  # close to numerical zero of the computer in double precision arithmetic

class OptimizationAccuracy(str, Enum):
    Sloppy = "Sloppy"  # very weak convergence
    Loose = "Loose"  # still weak convergence
    Medium = "Medium"  # intermediate accuracy
    Strong = "Strong"  # stronger
    Tight = "Tight"  # still stronger
    VeryTight = "VeryTight"  # even stronger
    Extreme = "Extreme"  # close to numerical zero of the computer in double precision arithmetic

class PrintLevel(str, Enum):
    SILENT = "SILENT"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class GridType(str, Enum):
    """Enum for DFT grid quality levels"""
    Grid1 = "Grid1"  # coarse grid
    Grid2 = "Grid2"  # standard grid
    Grid3 = "Grid3"  # fine grid
    Grid4 = "Grid4"  # very fine grid
    Grid5 = "Grid5"  # ultra fine grid

class SolventModel(str, Enum):
    """Enum for solvent models"""
    CPCM = "CPCM"
    SMD = "SMD"
    COSMO = "COSMO"
    COSMORS = "COSMO-RS"

@simstack_model
class ElProp(EmbeddedModel):
    """Explicit model for electrical property toggles.

    Defining this as a BaseModel ensures the generated JSON schema exposes
    fixed boolean properties (rendered as tickboxes) with the specified
    defaults instead of an open key/value editor.
    """
    model_config = {"extra": "forbid"}
    Dipole: bool = Field(False, json_schema_extra={"description": "Calculate dipole moment"})
    Quadrupole: bool = Field(False, json_schema_extra={"description": "Calculate quadrupole moment"})
    Polar: bool = Field(False, json_schema_extra={"description": "Calculate polarizability"})
    Hyperpol: bool = Field(False, json_schema_extra={"description": "Calculate hyperpolarizability"})
    HyperpolFrequencynm: float = Field(0.0, json_schema_extra={
        "description": "Frequency for hyperpolarizability calculation in nm"})
    PolarVelocity: bool = Field(False,
                                json_schema_extra={"description": "Calculate polarizability using velocity gauge"})
    PolarDipQuad: bool = Field(False, json_schema_extra={"description": "Calculate dipole-quadrupole polarizability"})
    PolarQuadQuad: bool = Field(False,
                                json_schema_extra={"description": "Calculate quadrupole-quadrupole polarizability"})

    @classmethod
    def json_schema(cls):

        schema = {
            "type": "object",
            "properties": {
                "compute_properties": {
                    "type": "boolean",
                    "default": False,
                    "title": "Compute properties",
                    "description": "Calculate electrical properties"
                }
            },
            "dependencies": {
                "compute_properties": {
                    "oneOf": [
                        {
                            "properties": {
                                "compute_properties": {"const": False}
                            }
                        },
                        {
                            "properties": {
                                "compute_properties": {"const": True},
                                **cls.model_json_schema().get('properties', {})
                            }
                        }
                    ]
                }
            }
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = {
            "compute_properties": {
                "ui:widget": "checkbox",
                "ui:title": "Compute properties"
            }
        }
        # Add conditions for all elprop fields
        elprop_fields = ["Dipole", "Quadrupole", "Polar", "Hyperpol", "HyperpolFrequencynm", "PolarVelocity", "PolarDipQuad", "PolarQuadQuad"]
        for field in elprop_fields:
            ui_schema[field] = {
                "ui:condition": {
                    "compute_properties": True
                }
            }
        # HyperpolFrequencynm also depends on Hyperpol
        ui_schema["HyperpolFrequencynm"]["ui:condition"] = {
            "compute_properties": True,
            "Hyperpol": True
        }
        return ui_schema

@simstack_model
class ExcitedStatesInput(EmbeddedModel):
    """Parameters for excited state calculations."""
    model_config = {"extra": "forbid"}
    states: int = Field(0, json_schema_extra={"description": "number of states to calculate, zero for ground state only"})
    focus_state: int = Field(1, json_schema_extra={"description": "state of focus"})
    active_electrons: int = Field(0, json_schema_extra={"description": "number of active electrons"})
    active_orbitals: int = Field(0, json_schema_extra={"description": "number of active orbitals"})

    @classmethod
    def json_schema(cls):
        schema = {
            "type": "object",
            "properties": {
                "excited_states": {
                    "type": "boolean",
                    "default": False,
                    "title": "Calculate excited states",
                    "description": "Calculate excited states"
                }
            },
            "dependencies": {
                "excited_states": {
                    "oneOf": [
                        {
                            "properties": {
                                "excited_states": {"const": False}
                            }
                        },
                        {
                            "properties": {
                                "excited_states": {"const": True},
                                **cls.model_json_schema().get('properties', {})
                            }
                        }
                    ]
                }
            }
        }
        # Remove model specific fields from the top level object if they were added
        # (though here we are constructing it from scratch)
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = {
            "excited_states": {
                "ui:widget": "checkbox",
                "ui:title": "Calculate excited states"
            }
        }

        ui_schema["states"] = {
            "ui:condition": {
                "excited_states": True
            }
        }
        ui_schema["focus_state"] = {
            "ui:condition": {
                "excited_states": True
            }
        }
        ui_schema["active_orbitals"] = {
            "ui:condition": {
                "excited_states": True,
                "method": ["CASSCF", "DFTMRCI"]
            }
        }
        ui_schema["active_electrons"] = {
            "ui:condition": {
                "excited_states": True,
                "method": ["CASSCF", "DFTMRCI"]
            }
        }

        return ui_schema



@simstack_model
class QMInput(Model):
    """
    A superclass for reusable parameters for quantum mechanical calculations.
    """
    model_config = {"extra": "ignore"}
    field_name: str = "QMInput"

    molecule: Molecule = Reference()
    charge: int = Field(0, json_schema_extra={"description": "net charge of the molecule"})
    multiplicity: int = Field(1,json_schema_extra={"description": "singlet,triplet,....."})
    open_shell_calculation: bool = Field(False, json_schema_extra={"description": "Open shell calculation"})
    basis_set: BasisSet
    functional: Functional
    method: QMMethod = Field(QMMethod.DFT, json_schema_extra={"description": "Quantum chemistry calculation method"})
    gradients: bool = Field(False, json_schema_extra={
        "description": "Calculate gradients (forces) for the molecule"})
    optimization: bool = Field(False, json_schema_extra={
        "description": "Perform geometry optimization"})
    frequencies: bool = Field(False, json_schema_extra={
        "description": "Calculate frequencies"})

    solvent: str = "None"
    solvent_model: SolventModel = Field(SolventModel.CPCM, json_schema_extra={"description": "Solvent model to use"})

    print_level: int = Field(1, json_schema_extra={ "description": "Print level for the calculation, 0-4"})
    scf_accuracy: SCFAccuracy = Field(SCFAccuracy.Medium, json_schema_extra={"description": "SCF convergence accuracy"})
    optimization_accuracy: OptimizationAccuracy = Field(OptimizationAccuracy.Medium, json_schema_extra={"description": "Geometry optimization accuracy"})
    grid_type: GridType = Field(GridType.Grid2, json_schema_extra={"description": "DFT grid quality level"})
    grid_spacing: float = Field(0.2, json_schema_extra={"description": "Grid spacing for the DFT calculation"})
    max_scf_iterations: int = Field(100, json_schema_extra={"description": "Maximum number of SCF iterations"})
    max_optimization_iterations: int = Field(100, json_schema_extra={"description": "Maximum number of geometry optimization iterations"})
    first_line: Optional[str] = Field("", json_schema_extra={"description": "additional input to the line of the input file"})
    blocks: List[str] = Field(default_factory=list, json_schema_extra={
        "description": "Additional blocks of text to be included in the input file, e.g. for constraints or custom settings"
    })

    # Electrical properties - ARE NOW MOVED TO TOP_LEVEL BY PROF. W _ SO NO LONGER EMBEDDED!
    Dipole: bool = Field(False, json_schema_extra={"description": "Calculate dipole moment"})
    Quadrupole: bool = Field(False, json_schema_extra={"description": "Calculate quadrupole moment"})
    Polar: bool = Field(False, json_schema_extra={"description": "Calculate polarizability"})
    Hyperpol: bool = Field(False, json_schema_extra={"description": "Calculate hyperpolarizability"})
    HyperpolFrequencynm: float = Field(0.0, json_schema_extra={
        "description": "Frequency for hyperpolarizability calculation in nm"})
    PolarVelocity: bool = Field(False,
                                json_schema_extra={"description": "Calculate polarizability using velocity gauge"})
    PolarDipQuad: bool = Field(False, json_schema_extra={"description": "Calculate dipole-quadrupole polarizability"})
    PolarQuadQuad: bool = Field(False,
                                json_schema_extra={"description": "Calculate quadrupole-quadrupole polarizability"})

    # Excited states parameters
    states: int = Field(0, json_schema_extra={"description": "number of states to calculate, zero for ground state only"})
    focus_state: int = Field(1, json_schema_extra={"description": "state of focus"})
    active_electrons: int = Field(0, json_schema_extra={"description": "number of active electrons"})
    active_orbitals: int = Field(0, json_schema_extra={"description": "number of active orbitals"})

    restart_files: FileList = Field(default_factory=FileList, json_schema_extra={
         "description": "Files to be used for restarting the calculation"
    })
    tolerate_failure: bool = Field(False, json_schema_extra={"description": "Tolerate failure of the calculation"})

    # UI persistence toggles (real fields so they are saved to the DB)
    excited_states: bool = Field(False, json_schema_extra={"title": "Calculate excited states"})
    compute_properties: bool = Field(False, json_schema_extra={"title": "Compute properties"})
    use_solvent: bool = Field(False, json_schema_extra={"title": "Use solvent"})
    non_standard_inputs: bool = Field(False, json_schema_extra={"title": "Non-standard inputs"})
    non_standard_parameters: bool = Field(False, json_schema_extra={"title": "Non-standard parameters"})

    @model_validator(mode="before")
    @classmethod
    def validate_before(cls, data):
        """Ensure fieldname is set for existing documents and handle null embedded models"""
        if isinstance(data, dict):
            if "field_name" not in data:
                data["field_name"] = cls.__name__

            # Ensure embedded models are never None if present as None in DB
            # Also add missing keys with empty dict to satisfy odmantic
            if "restart_files" not in data or data.get("restart_files") is None:
                data["restart_files"] = {}
        return data


    @model_validator(mode='before')
    @classmethod
    def set_default_solvent(cls, data) -> dict:
        """Set default solvent to 'None' if not provided or is None"""
        if isinstance(data, dict):
            if "solvent" not in data or data["solvent"] is None:
                data["solvent"] = "None"
            if "frequencies" not in data or data["frequencies"] is None:
                data["frequencies"] = False

            # Infer toggle states for existing records if missing
            if "use_solvent" not in data:
                data["use_solvent"] = data.get("solvent", "None") != "None"

            # Handle legacy elprop and excited_states_input
            elprop_data = data.pop("elprop", {})
            if isinstance(elprop_data, dict):
                for k, v in elprop_data.items():
                    if k not in data:
                        data[k] = v
            elif hasattr(elprop_data, "model_dump"):
                for k, v in elprop_data.model_dump().items():
                    if k not in data:
                        data[k] = v

            es_input_data = data.pop("excited_states_input", {})
            if isinstance(es_input_data, dict):
                for k, v in es_input_data.items():
                    if k not in data:
                        data[k] = v
            elif hasattr(es_input_data, "model_dump"):
                for k, v in es_input_data.model_dump().items():
                    if k not in data:
                        data[k] = v

            if "excited_states" not in data:
                data["excited_states"] = data.get("states", 0) > 0

            if "compute_properties" not in data:
                elprop_fields = ["Dipole", "Quadrupole", "Polar", "Hyperpol", "PolarVelocity", "PolarDipQuad", "PolarQuadQuad"]
                data["compute_properties"] = any(data.get(k) for k in elprop_fields) or data.get("HyperpolFrequencynm", 0) > 0

            # If toggles are False, reset fields to default
            if not data.get("compute_properties"):
                elprop_fields = ["Dipole", "Quadrupole", "Polar", "Hyperpol", "PolarVelocity", "PolarDipQuad", "PolarQuadQuad"]
                for f in elprop_fields:
                    data[f] = False
                data["HyperpolFrequencynm"] = 0.0

            if not data.get("excited_states"):
                for f in ["states", "focus_state", "active_electrons", "active_orbitals"]:
                    if f == "states":
                        data[f] = 0
                    elif f == "focus_state":
                        data[f] = 1
                    else:
                        data[f] = 0

            # Map use_solvent to solvent
            if not data.get("use_solvent"):
                data["solvent"] = "None"

        return data

    @model_validator(mode='after')
    def validate_calculation_options(self) -> Self:
        """Sanity checks for mutually incompatible options."""
        # Frequencies (aoforce) are currently implemented only for ground-state jobs.
        if self.frequencies and self.states > 0:
            raise ValueError("Frequency calculations for excited states (states > 0) are not supported. Set states=0 or frequencies=False.")

        # Hyperpolarizability is currently implemented only for ground-state jobs.
        if self.Hyperpol and self.states > 0:
            raise ValueError("Hyperpolarizability for excited states (states > 0) is not supported. Set states=0.")

        if self.HyperpolFrequencynm < 0:
            raise ValueError("HyperpolFrequencynm must be >= 0 (0 = static).")

        return self

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def elprop(self) -> ElProp:
        """Return an :class:`ElProp` view constructed from top-level fields.

        Historically, QMInput exposed an embedded ``elprop`` submodel which
        downstream helpers (such as the ORCA input builders) used directly.
        The new QMInput layout stores individual electrical-property toggles
        (Dipole, Quadrupole, ...) as top-level fields plus a
        ``compute_properties`` flag. This property provides a backwards-
        compatible view so existing code that accesses ``qm_input.elprop``
        continues to work without requiring the DB schema to change again.
        """

        return ElProp(
            Dipole=self.Dipole,
            Quadrupole=self.Quadrupole,
            Polar=self.Polar,
            Hyperpol=self.Hyperpol,
            HyperpolFrequencynm=self.HyperpolFrequencynm,
            PolarVelocity=self.PolarVelocity,
            PolarDipQuad=self.PolarDipQuad,
            PolarQuadQuad=self.PolarQuadQuad,
        )
    @classmethod
    def json_schema(cls, recursive=True):
        """
        Generate a JSON schema for this model.

        :param recursive: If True, use recursive strategy that integrates all child schemas.
                         If False, use non-recursive strategy that allows frontend to handle embedded models.
        :return: A JSON schema dictionary.
        """
        schema = cleaned_json_schema(cls)
        schema['title'] = cls.__name__

        # delete gradients from schema if it exists (it's handled by optimization dependency)
        if 'gradients' in schema['properties']:
            del schema['properties']['gradients']

        # Force type to string for first_line to avoid dropdown for Optional[str]
        if 'first_line' in schema['properties']:
            first_line_schema = schema['properties']['first_line']
            if 'anyOf' in first_line_schema:
                first_line_schema['type'] = 'string'
                del first_line_schema['anyOf']

        # Extract and remove schemas for conditional fields from top-level properties
        prop_schemas = schema['properties']

        # Method-dependent functional
        functional_schema = prop_schemas.pop('functional', None)

        # Optimization-dependent fields
        opt_acc_schema = prop_schemas.pop('optimization_accuracy', None)

        # Elprop fields
        elprop_fields = ["Dipole", "Quadrupole", "Polar", "Hyperpol", "HyperpolFrequencynm", "PolarVelocity", "PolarDipQuad", "PolarQuadQuad"]
        elprop_schemas = {field: prop_schemas.pop(field) for field in elprop_fields if field in prop_schemas}

        # Excited state fields
        es_fields = ["states", "focus_state", "active_electrons", "active_orbitals"]
        es_schemas = {field: prop_schemas.pop(field) for field in es_fields if field in prop_schemas}

        # Non-standard inputs
        nsi_fields = ['first_line', 'blocks', 'restart_files']
        nsi_schemas = {field: prop_schemas.pop(field) for field in nsi_fields if field in prop_schemas}

        # Non-standard parameters
        nsp_fields = ['print_level', 'scf_accuracy', 'grid_type', 'grid_spacing', 'max_scf_iterations', 'max_optimization_iterations']
        nsp_schemas = {field: prop_schemas.pop(field) for field in nsp_fields if field in prop_schemas}

        # Solvent fields
        solvent_schema = prop_schemas.pop('solvent', None)
        solvent_model_schema = prop_schemas.pop('solvent_model', None)

        if 'dependencies' not in schema:
            schema['dependencies'] = {}

        schema['dependencies'].update({
            "compute_properties": {
                "oneOf": [
                    {
                        "properties": {
                            "compute_properties": {"const": False}
                        }
                    },
                    {
                        "properties": {
                            "compute_properties": {"const": True},
                            **elprop_schemas
                        }
                    }
                ]
            },
            "excited_states": {
                "oneOf": [
                    {
                        "properties": {
                            "excited_states": {"const": False}
                        }
                    },
                    {
                        "properties": {
                            "excited_states": {"const": True},
                            **es_schemas
                        }
                    }
                ]
            },
            "use_solvent": {
                "oneOf": [
                    {
                        "properties": {
                            "use_solvent": {"const": False}
                        }
                    },
                    {
                        "properties": {
                            "use_solvent": {"const": True},
                            "solvent": solvent_schema,
                            "solvent_model": solvent_model_schema
                        }
                    }
                ]
            },
            "non_standard_inputs": {
                "oneOf": [
                    {
                        "properties": {
                            "non_standard_inputs": {"const": False}
                        }
                    },
                    {
                        "properties": {
                            "non_standard_inputs": {"const": True},
                            **nsi_schemas
                        }
                    }
                ]
            },
            "non_standard_parameters": {
                "oneOf": [
                    {
                        "properties": {
                            "non_standard_parameters": {"const": False}
                        }
                    },
                    {
                        "properties": {
                            "non_standard_parameters": {"const": True},
                            **nsp_schemas
                        }
                    }
                ]
            },
            "method": {
                "oneOf": [
                    {
                        "properties": {
                            "method": {"enum": ["CASSCF", "DFTMRCI"]}
                        }
                    },
                    {
                        "properties": {
                            "method": {"enum": ["DFT", "TDDFT"]},
                            "functional": functional_schema
                        },
                        "required": ["functional"] if functional_schema else []
                    },
                    {
                        "properties": {
                            "method": {
                                "not": {"enum": ["CASSCF", "DFTMRCI", "DFT", "TDDFT"]}
                            }
                        }
                    }
                ]
            },
            "optimization": {
                "oneOf": [
                    {
                        "properties": {
                            "optimization": {"const": False},
                            "gradients": {"type": "boolean"},
                        }
                    },
                    {
                        "properties": {
                            "optimization": {"const": True},
                            "optimization_accuracy": opt_acc_schema
                        }
                    }
                ]
            }
        })

        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)

        ui_schema["field_name"] = {"ui:widget": "hidden"}

        # non_standard_inputs toggle
        ui_schema["non_standard_inputs"] = {
            "ui:widget": "checkbox",
            "ui:title": "Non-standard inputs"
        }

        # non_standard_parameters toggle
        ui_schema["non_standard_parameters"] = {
            "ui:widget": "checkbox",
            "ui:title": "Non-standard parameters"
        }

        # use_solvent toggle
        ui_schema["use_solvent"] = {
            "ui:widget": "checkbox",
            "ui:title": "Use solvent"
        }

        # compute_properties toggle
        ui_schema["compute_properties"] = {
            "ui:widget": "checkbox",
            "ui:title": "Compute properties"
        }

        # Add conditions for elprop fields
        elprop_fields = ["Dipole", "Quadrupole", "Polar", "Hyperpol", "HyperpolFrequencynm", "PolarVelocity", "PolarDipQuad", "PolarQuadQuad"]
        for field in elprop_fields:
            ui_schema[field] = {
                "ui:condition": {
                    "compute_properties": True
                }
            }
        # HyperpolFrequencynm also depends on Hyperpol
        ui_schema["HyperpolFrequencynm"]["ui:condition"] = {
            "compute_properties": True,
            "Hyperpol": True
        }

        # excited_states toggle
        ui_schema["excited_states"] = {
            "ui:widget": "checkbox",
            "ui:title": "Calculate excited states"
        }

        # Add conditions for excited state fields
        for field in ["states", "focus_state", "active_orbitals", "active_electrons"]:
            ui_schema[field] = {
                "ui:condition": {
                    "excited_states": True
                }
            }

        # focus_state additionally depends on method
        ui_schema["focus_state"]["ui:condition"] = {
            "excited_states": True,
            "method": ["CASSCF", "DFTMRCI"]
        }

        # active_orbitals and active_electrons also depend on method
        ui_schema["active_orbitals"]["ui:condition"] = {
            "excited_states": True,
            "method": ["CASSCF", "DFTMRCI"]
        }
        ui_schema["active_electrons"]["ui:condition"] = {
            "excited_states": True,
            "method": ["CASSCF", "DFTMRCI"]
        }

        ui_schema["optimization_accuracy"] = {
            "ui:condition": {
                "optimization": True
            }
        }

        ui_schema["first_line"] = {
            "ui:condition": {
                "non_standard_inputs": True
            }
        }

        ui_schema["blocks"] = {
            "ui:condition": {
                "non_standard_inputs": True
            }
        }

        ui_schema["restart_files"] = {
            "ui:condition": {
                "non_standard_inputs": True
            }
        }

        ui_schema["solvent"] = {
            "ui:condition": {
                "use_solvent": True
            }
        }

        ui_schema["solvent_model"] = {
            "ui:condition": {
                "use_solvent": True
            },
        }

        ui_schema["print_level"] = {
            "ui:condition": {
                "non_standard_parameters": True
            }
        }
        ui_schema["scf_accuracy"] = {
            "ui:condition": {
                "non_standard_parameters": True
            }
        }
        ui_schema["grid_type"] = {
            "ui:condition": {
                "non_standard_parameters": True
            }
        }
        ui_schema["grid_spacing"] = {
            "ui:condition": {
                "non_standard_parameters": True
            }
        }
        ui_schema["max_scf_iterations"] = {
            "ui:condition": {
                "non_standard_parameters": True
            }
        }
        ui_schema["max_optimization_iterations"] = {
            "ui:condition": {
                "non_standard_parameters": True
            }
        }


        ui_schema["functional"] = {
            "ui:condition": {
                "method": {
                    "ui:options": ["DFT", "TDDFT"]
                }
            }
        }

        ui_schema["ui:order"] = [
            "molecule",
            "name",
            "non_standard_inputs",
            "first_line",
            "blocks",
            "restart_files",
            "charge",
            "multiplicity",
            "open_shell_calculation",
            "method",
            "functional",
            "basis_set",
            "excited_states",
            "states",
            "focus_state",
            "active_orbitals",
            "active_electrons",
            "use_solvent",
            "solvent",
            "solvent_model",
            "optimization",
            "gradients",
            "frequencies",
            "compute_properties",
            "Dipole",
            "Quadrupole",
            "Polar",
            "Hyperpol",
            "HyperpolFrequencynm",
            "PolarVelocity",
            "PolarDipQuad",
            "PolarQuadQuad",
            "non_standard_parameters",
            "print_level",
            "scf_accuracy",
            "grid_type",
            "grid_spacing",
            "max_scf_iterations",
            "max_optimization_iterations",
            "id",
            "tolerate_failure",
            ]
        if "ui:options" not in ui_schema:
            ui_schema["ui:options"] = {}
        ui_schema["ui:options"]["ui:foldable"] = True

        return ui_schema


@simstack_model
class DummyQMInput(Model):
    """QM settings *without* a molecule.

    This model is intended to mirror :class:`QMInput` as closely as
    possible, but omits the ``molecule`` reference so it can be used as a
    template in workflows where geometries are supplied by other nodes
    (e.g. conformer generation). It is primarily consumed by the ORCA
    workflow helpers in
    ``applications.electronic_structure.orca.many_orca_jobs_from_smiles``.
    """

    model_config = {"extra": "ignore"}
    field_name: str = "DummyQMInput"

    # Note: no ``molecule`` field here on purpose.
    charge: int = Field(0, json_schema_extra={"description": "net charge of the molecule"})
    multiplicity: int = Field(1, json_schema_extra={"description": "singlet,triplet,....."})
    open_shell_calculation: bool = Field(False, json_schema_extra={"description": "Open shell calculation"})

    basis_set: BasisSet
    functional: Functional
    method: QMMethod = Field(QMMethod.DFT, json_schema_extra={"description": "Quantum chemistry calculation method"})

    gradients: bool = Field(False, json_schema_extra={
        "description": "Calculate gradients (forces) for the molecule"})
    optimization: bool = Field(False, json_schema_extra={
        "description": "Perform geometry optimization"})
    frequencies: bool = Field(False, json_schema_extra={
        "description": "Calculate frequencies"})

    solvent: str = "None"
    solvent_model: SolventModel = Field(SolventModel.CPCM, json_schema_extra={"description": "Solvent model to use"})

    print_level: int = Field(1, json_schema_extra={"description": "Print level for the calculation, 0-4"})
    scf_accuracy: SCFAccuracy = Field(SCFAccuracy.Medium, json_schema_extra={"description": "SCF convergence accuracy"})
    optimization_accuracy: OptimizationAccuracy = Field(OptimizationAccuracy.Medium, json_schema_extra={"description": "Geometry optimization accuracy"})
    grid_type: GridType = Field(GridType.Grid2, json_schema_extra={"description": "DFT grid quality level"})
    grid_spacing: float = Field(0.2, json_schema_extra={"description": "Grid spacing for the DFT calculation"})
    max_scf_iterations: int = Field(100, json_schema_extra={"description": "Maximum number of SCF iterations"})
    max_optimization_iterations: int = Field(100, json_schema_extra={"description": "Maximum number of geometry optimization iterations"})
    first_line: Optional[str] = Field("", json_schema_extra={"description": "additional input to the line of the input file"})
    blocks: List[str] = Field(default_factory=list, json_schema_extra={
        "description": "Additional blocks of text to be included in the input file, e.g. for constraints or custom settings"
    })

    # Electrical properties – mirror QMInput top-level flags
    Dipole: bool = Field(False, json_schema_extra={"description": "Calculate dipole moment"})
    Quadrupole: bool = Field(False, json_schema_extra={"description": "Calculate quadrupole moment"})
    Polar: bool = Field(False, json_schema_extra={"description": "Calculate polarizability"})
    Hyperpol: bool = Field(False, json_schema_extra={"description": "Calculate hyperpolarizability"})
    HyperpolFrequencynm: float = Field(
        0.0,
        json_schema_extra={
            "description": "Frequency for hyperpolarizability calculation in nm",
        },
    )
    PolarVelocity: bool = Field(
        False,
        json_schema_extra={
            "description": "Calculate polarizability using velocity gauge",
        },
    )
    PolarDipQuad: bool = Field(
        False,
        json_schema_extra={"description": "Calculate dipole-quadrupole polarizability"},
    )
    PolarQuadQuad: bool = Field(
        False,
        json_schema_extra={"description": "Calculate quadrupole-quadrupole polarizability"},
    )

    # Excited states parameters – same shape as QMInput
    states: int = Field(
        0,
        json_schema_extra={
            "description": "number of states to calculate, zero for ground state only",
        },
    )
    focus_state: int = Field(1, json_schema_extra={"description": "state of focus"})
    active_electrons: int = Field(
        0,
        json_schema_extra={"description": "number of active electrons"},
    )
    active_orbitals: int = Field(
        0,
        json_schema_extra={"description": "number of active orbitals"},
    )

    # Legacy hyperpolarizability frequency knobs used by workflow drivers.
    # These remain for compatibility with existing DummyQMInput templates
    # and are validated below in ``validate_hyperpol_frequencies``.
    hyperpol_frequency_nm: float = Field(
        0.0,
        json_schema_extra={
            "description": "Wavelength in nm for dynamic hyperpolarizability (0 = static/DC)",
        },
    )
    polar_imag_frequency_nm: Optional[float] = Field(
        None,
        json_schema_extra={
            "description": "Imaginary frequency component in nm for dynamic hyperpolarizability",
        },
    )

    restart_files: FileList = Field(
        default_factory=FileList,
        json_schema_extra={"description": "Files to be used for restarting the calculation"},
    )
    tolerate_failure: bool = Field(
        False,
        json_schema_extra={"description": "Tolerate failure of the calculation"},
    )

    # UI persistence toggles – mirrored for completeness; in practice the
    # DummyQMInput is usually not exposed directly in the GUI.
    excited_states: bool = Field(
        False,
        json_schema_extra={"title": "Calculate excited states"},
    )
    compute_properties: bool = Field(
        False,
        json_schema_extra={"title": "Compute properties"},
    )
    use_solvent: bool = Field(False, json_schema_extra={"title": "Use solvent"})
    non_standard_inputs: bool = Field(
        False,
        json_schema_extra={"title": "Non-standard inputs"},
    )
    non_standard_parameters: bool = Field(
        False,
        json_schema_extra={"title": "Non-standard parameters"},
    )

    # ------------------------------------------------------------------
    # Derived views (match QMInput)
    # ------------------------------------------------------------------

    @property
    def elprop(self) -> ElProp:
        """Return an :class:`ElProp` view constructed from top-level flags.

        This mirrors :meth:`QMInput.elprop` so existing workflow helpers
        that access ``dummy_qm_input.elprop`` continue to work while the
        actual storage lives in the top-level boolean fields.
        """

        return ElProp(
            Dipole=self.Dipole,
            Quadrupole=self.Quadrupole,
            Polar=self.Polar,
            Hyperpol=self.Hyperpol,
            HyperpolFrequencynm=self.HyperpolFrequencynm,
            PolarVelocity=self.PolarVelocity,
            PolarDipQuad=self.PolarDipQuad,
            PolarQuadQuad=self.PolarQuadQuad,
        )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure field_name is set for existing documents."""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @model_validator(mode="before")
    @classmethod
    def set_default_restart_files(cls, data) -> dict:
        """Set default restart_files to empty FileList if not provided or is None."""
        if isinstance(data, dict):
            if "restart_files" not in data or data["restart_files"] is None:
                data["restart_files"] = FileList()
        return data

    @model_validator(mode="before")
    @classmethod
    def set_default_solvent(cls, data) -> dict:
        """Set default solvent to 'None' if not provided or is None."""
        if isinstance(data, dict):
            if "solvent" not in data or data["solvent"] is None:
                data["solvent"] = "None"
            if "frequencies" not in data or data["frequencies"] is None:
                data["frequencies"] = False
        return data

    @model_validator(mode="after")
    def validate_hyperpol_frequencies(self) -> Self:
        """Sanity checks for hyperpolarizability frequency inputs.

        This mirrors the legacy behaviour from the old QMInput/DummyQMInput
        models so that existing templates continue to validate.
        """

        hyperpol_set = self.hyperpol_frequency_nm is not None and self.hyperpol_frequency_nm != 0.0
        polar_imag_set = self.polar_imag_frequency_nm is not None and self.polar_imag_frequency_nm != 0.0

        if hyperpol_set and polar_imag_set:
            raise ValueError(
                "Please specify only one of hyperpol_frequency_nm or polar_imag_frequency_nm; not both."
            )

        if self.hyperpol_frequency_nm is not None and self.hyperpol_frequency_nm < 0:
            raise ValueError("hyperpol_frequency_nm must be >= 0 (0 = static).")
        if self.polar_imag_frequency_nm is not None and self.polar_imag_frequency_nm < 0:
            raise ValueError("polar_imag_frequency_nm must be >= 0 (0 = static).")

        return self

    @classmethod
    def json_schema(cls, recursive: bool = True):  # pragma: no cover - thin wrapper
        """Generate a cleaned JSON schema for the DummyQMInput model."""
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        return schema

    @classmethod
    def ui_schema(cls):  # pragma: no cover - thin wrapper
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        if "ui:options" not in ui_schema:
            ui_schema["ui:options"] = {}
        ui_schema["ui:options"]["ui:foldable"] = True
        return ui_schema
        if "ui:options" not in ui_schema:
            ui_schema["ui:options"] = {}
        ui_schema["ui:options"]["ui:foldable"] = True
        return ui_schema
