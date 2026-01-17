from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RealEstateType(str, Enum):
    enebolig = "enebolig"
    tomannsbolig = "tomannsbolig"
    rekkehus = "rekkehus"
    leilighet = "leilighet"
    næringseiendom = "næringseiendom"
    hytte = "hytte"


class EstimationFeatures(BaseModel):
    """
    Strict input contract for estimation.

    We prefer to return 422 rather than producing a misleading estimate.
    """
    model_config = ConfigDict(extra="ignore")

    # Required categorical / geo
    realestate_type: RealEstateType = Field(...)
    municipality_number: int = Field(..., ge=1)

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)

    # Required numeric
    built_year: int = Field(..., ge=1800, le=2100)

    # Areas are required (both)
    total_area: float = Field(..., gt=0)
    bra: float = Field(..., gt=0)  # external name from upstream (usable area)

    # Optional features
    floor: Optional[int] = None
    bedrooms: Optional[int] = Field(default=None, ge=0)
    rooms: Optional[int] = Field(default=None, ge=0)

    gnr_number: Optional[int] = None
    bnr_number: Optional[int] = None
    snr_number: Optional[int] = None

    @property
    def usable_area(self) -> float:
        # internal rename: bra -> usable_area
        return self.bra

    @model_validator(mode="after")
    def strict_validation(self):
        # Domain sanity checks
        if self.total_area < self.bra:
            raise ValueError("total_area must be >= bra")

        # Conditional required fields
        if self.realestate_type == RealEstateType.leilighet and self.floor is None:
            raise ValueError("floor is required for realestate_type 'leilighet'")

        return self


EstimateRequest = Dict[str, EstimationFeatures]


class EstimateResult(BaseModel):
    estimated_price: int
    currency: str = "NOK"
    model_version: str = "stub-v1"
    warnings: list[str] = Field(default_factory=list)


EstimateResponse = Dict[str, EstimateResult]
