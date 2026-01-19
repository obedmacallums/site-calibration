from pydantic import BaseModel, Field
import typing

class HorizontalParameters(BaseModel):
    a: float
    b: float
    tE: float
    tN: float

class VerticalParameters(BaseModel):
    vertical_shift: float
    slope_north: float
    slope_east: float
    centroid_north: float
    centroid_east: float

class CalibrationParameters(BaseModel):
    horizontal: HorizontalParameters
    vertical: VerticalParameters

class ResidualPoint(BaseModel):
    Point: str
    dE: float
    dN: float
    dH: float

class CalibrationResult(BaseModel):
    parameters: CalibrationParameters
    residuals: typing.List[ResidualPoint]
    report: str
