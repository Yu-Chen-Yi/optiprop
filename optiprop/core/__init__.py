"""Canonical domain objects for OptiProp 2.0."""

from .field import EX_EY_COMPONENTS, SCALAR_COMPONENTS, Field2D
from .grid import Grid2D
from .validation import Severity, ValidationError, ValidationIssue, ValidationReport

__all__ = [
    "EX_EY_COMPONENTS",
    "Field2D",
    "Grid2D",
    "SCALAR_COMPONENTS",
    "Severity",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
]
