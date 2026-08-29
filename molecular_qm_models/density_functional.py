from enum import Enum
from typing import List, Dict, Any

from odmantic import Field, EmbeddedModel, Model
from pydantic import model_validator

from molecular_qm_models.dispersion_correction import DispersionCorrection, DispersionCorrectionEnum
from simstack.models import simstack_model


class FunctionalEnum(str,Enum):
    """
    Enumeration of common density functionals.
    """
    # Pure GGA Functionals
    PBE = "PBE"
    BLYP = "BLYP"
    BP86 = "BP86"
    
    # from orca
    #  BNULL      # Becke '88 exchange, no correlation
    #     BVWN       # Becke '88 exchange, VWN-5 correlation
    #     BP         # Becke '88 exchange, Perdew '86 correlation
    #     PW91       # Perdew-Wang (PW) GGA-II '91 functional
    #     mPWPW      # Modified PW exchange, PW correlation
    #     mPWLYP     # Modified PW exchange, Lee-Yang-Parr (LYP) correlation
    #     BLYP       # Becke '88 exchange, LYP correlation
    #     GP         # Gill '96 exchange, Perdew '86 correlation
    #     GLYP       # Gill '96 exchange, LYP correlation
    #     PBE        # Perdew-Burke-Ernzerhof (PBE) functional
    #     revPBE     # Revised PBE (exchange scaling)
    #     RPBE       # Revised PBE (modified exchange functional)
    #     PWP        # PW '91 exchange, Perdew '86 correlation
    #     OLYP       # Hoe/Cohen/Handy's optimized exchange, LYP correlation
    #     OPBE       # Hoe/Cohen/Handy's optimized exchange, PBE correlation
    #     XLYP       # Xu/Goddard exchange, LYP correlation
    #     B97D       # Grimme's GGA including D2 dispersion correction
    #     PW86PBE    # PW '86 exchange, PBE correlation (as used for vdw-DF and related)
    #     RPW86PBE   # Revised PW '86 exchange, PBE correlation
    #     
    BVWN = "BVWN"
    BP = "BP"
    PW91 = "PW91"
    mPWPW = "mPWPW"
    mPWLYP = "mPWLYP"
    GP = "GP"
    GLYP = "GLYP"
    revPBE = "revPBE"
    RPBE = "RPBE"
    PWP = "PWP"
    OLYP = "OLYP"
    OPBE = "OPBE"
    XLYP = "XLYP"
    B97D = "B97D"
    PW86PBE = "PW86PBE"
    RPW86PBE = "RPW86PBE"
    
    # Hybrid GGA Functionals
    B3LYP = "B3LYP"
    PBE0 = "PBE0"
    HSE06 = "HSE06"
    ###from the orca manual
    #***************************************
    # B1LYP      # 1-parameter hybrid of BLYP (25% HF exchange)
    # B1P        # Similar with Perdew '86 correlation
    # G1LYP      # 1-parameter analog with Gill '96 exchange
    # G1P        # Similar with Perdew '86 correlation
    # B3LYP      # 3-parameter Hybrid of BLYP (20% HF exchange)
    # B3P        # Similar with Perdew '86 correlation
    # G3LYP      # 3-parameter analog with Gill '96 exchange
    # G3P        # Similar with Perdew '86 correlation
    # PBE0       # 1-parameter version of PBE (25% HF exchange)
    # PWP1       # 1-parameter version of PWP (analog of PBE0)
    # mPW1PW     # 1-parameter version of mPWPW (analog of PBE0)
    # mPW1LYP    # 2-parameter version of mPWLYP (analog of PBE0)
    # PW91_0     # 1-parameter version of PW91 (analog of PBE0)
    # O3LYP      # 3-parameter version of OLYP 
    # X3LYP      # 3-parameter version of XLYP 
    # B97        # Becke's original hybrid functional
    # BHANDHLYP  # Becke's half-and-half hybrid functional (50% HF exchange)
    B1LYP = "B1LYP"
    B1P = "B1P"
    G1LYP = "G1LYP"
    G1P = "G1P"
    B3P = "B3P"
    G3LYP = "G3LYP"
    G3P = "G3P"
    PWP1 = "PWP1"
    mPW1PW = "mPW1PW"
    mPW1LYP = "mPW1LYP"
    PW91_0 = "PW91_0"
    O3LYP = "O3LYP"
    X3LYP = "X3LYP"
    B97 = "B97"
    BHANDHLYP = "BHANDHLYP"     

    
    # Meta-GGA Functionals
    TPSS = "TPSS"
    M06L = "M06-L"
    REVTPSS = "revTPSS"    # Revised TPSS functional
    SCANfunc = "SCANfunc"   # Perdew's SCAN functional
    RSCAN = "RSCAN"      # Regularized SCAN functional
    R2SCAN = "R2SCAN"     # Regularized and restored SCAN functional

    # Hybrid Meta-GGA Functionals
    TPSSh = "TPSSh"
    M06 = "M06"
    M062X = "M06-2X"
    
    # Double-Hybrid Functionals
    B2PLYP = "B2PLYP"
    
    # Range-Separated Hybrid Functionals
    CAM_B3LYP = "CAM-B3LYP"
    WB97XD = "wB97X-D" #which package was that for?# omega as non ascii replaced by w. should be correct for all our used packages
    WB97XD3BJ = "wB97X-D3BJ" #orca
    WB97XD3 = "wB97X-D3"#orca
    WB97X = "wB97X" #orca
    WB97= "wB97"#orca
    LCBLYP = "LC-BLYP" #orca
    LCPBE = "LC-PBE" #orca

    # Minnesota Functionals
    MN15 = "MN15"

    # Semiempirical methods
    GFN2xTB = "GFN2-xTB"
    GFN1xTB = "GFN1-xTB"
    MNDO = "MNDO"
    AM1 = "AM1"
    PM1 = "PM1"
    PM6 = "PM6"
    OM1 = "OM1"
    OM2 = "OM2"
    OM3 = "OM3"

    @classmethod
    def get_description(cls, functional):
        """
        Returns a description of the functional type and its characteristics.
        """
        descriptions = {
            cls.PBE: "Pure GGA functional, good for solid-state physics",
            cls.BLYP: "Pure GGA functional, popular in molecular chemistry",
            cls.B3LYP: "Hybrid functional (20% HF), standard for molecular chemistry",
            cls.PBE0: "Hybrid functional (25% HF), good for both molecules and solids",
            cls.HSE06: "Range-separated hybrid of PBE, good for solid-state",
            cls.TPSS: "Meta-GGA functional, improved accuracy over GGA",
            cls.M06L: "Meta-GGA functional, good for organometallics",
            cls.M062X: "Hybrid meta-GGA (54% HF), good for main group thermochemistry",
            cls.CAM_B3LYP: "Range-separated hybrid, good for excited states",
            cls.WB97XD: "Range-separated hybrid with empirical dispersion",
            cls.B2PLYP: "Double hybrid (53% MP2), high accuracy for small molecules"
        }
        return descriptions.get(functional, "No description available")

@simstack_model
class Functional(EmbeddedModel):
    """
    A class representing a density functional for quantum mechanical calculations.
    
    Attributes:
        functional: The density functional to be used

    """
    field_name: str = "Functional"
    functional: FunctionalEnum = Field(
        FunctionalEnum.B3LYP,
        json_schema_extra={
            "enum": [e.value for e in FunctionalEnum],
            "description": "density functional"
        }
    )
    
    dispersion_correction: DispersionCorrection = Field(default=DispersionCorrection(value=DispersionCorrectionEnum.NONE))

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    # @property
    # def functional(self) -> FunctionalEnum:
    #     """Get the functional enum value."""
    #     return FunctionalEnum(self.functional_str)
    #
    # @functional.setter
    # def functional(self, value: FunctionalEnum):
    #     """Set the functional enum value."""
    #     if not isinstance(value, FunctionalEnum):
    #         raise ValueError(f"Invalid functional: {value}")
    #     self.functional_str = value.value

    def get_functional_type(self) -> str:
        """
        Returns the type of functional (Pure GGA, Hybrid, Meta-GGA, etc.)
        """
        if self.functional in [FunctionalEnum.PBE, FunctionalEnum.BLYP, FunctionalEnum.BP86]:
            return "Pure GGA"
        elif self.functional in [FunctionalEnum.B3LYP, FunctionalEnum.PBE0]:
            return "Hybrid GGA"
        elif self.functional in [FunctionalEnum.TPSS, FunctionalEnum.M06L]:
            return "Meta-GGA"
        elif self.functional in [FunctionalEnum.CAM_B3LYP, FunctionalEnum.WB97XD]:
            return "Range-Separated Hybrid"
        elif self.functional == FunctionalEnum.B2PLYP:
            return "Double Hybrid"
        return "Unknown"

    def get_description(self) -> str:
        """
        Returns a description of the functional including its characteristics
        """
        return FunctionalEnum.get_description(self.functional)

    def __str__(self) -> str:
        base = self.functional.value + str(self.dispersion_correction)

        return base


    def make_table_entries(self,**kwargs):
        return {'functional': str(self.functional.value), 'dispersion': str(self.dispersion_correction.value.value)}

    def make_column_defs_instance(self,**kwargs) -> List[Dict[str, Any]]:
        field_prefix = kwargs.get("field_prefix", "")
        return [{'headerName': 'Functional', 'field': field_prefix + 'functional' },
                {'headerName': 'Disp. Corr.', 'field': field_prefix + 'dispersion' }, ]


@simstack_model
class FunctionalModel(Model):
    """
    A model representing a density functional with optional dispersion correction.

    Attributes:
        functional: The density functional to be used

    """

    field_name: str = "FunctionalModel"
    functional: Functional = Field(default=Functional(functional=FunctionalEnum.B3LYP, dispersion_correction=DispersionCorrection()))

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        """Ensure fieldname is set for existing documents"""
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'FunctionalModel':
        """
        Create a FunctionalModel instance from a dictionary.
        """
        return cls(**data)

if __name__ == "__main__":
    # Example usage
    dispersion = DispersionCorrection(
        use_dispersion_correction=True,
        version=DispersionCorrectionEnum.D3,
        include_three_body=True,
        cutoff_radius=1.2,
        scale_factors={"s6": 1.0, "s8": 1.0, "sr6": 1.0, "sr8": 1.0, "alpha6": 14.0},
        custom_parameters={"custom_param": "value"}
    )
    functional = Functional(functional=FunctionalEnum.B3LYP, dispersion_correction = dispersion)
    print(functional.get_description())
    print(functional.get_functional_type())
    print(str(functional))
    #dict_repr = functional.model_dump()
    dict_repr =  {'functional': FunctionalEnum.PBE, 'dispersion_correction': dispersion.model_dump()}
    print(dict_repr)
    new_functional = Functional.from_dict(dict_repr)
    print(new_functional)
