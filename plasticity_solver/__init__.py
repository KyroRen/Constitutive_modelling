"""
Plasticity solver package for J2 plasticity with kinematic hardening.

Provides material models (Chaboche, Ohno-Wang II, Abdel-Karim-Ohno, Kang, etc.)
and unified uniaxial/multi-axial solvers with explicit, IMEX, and fully-implicit methods.
"""

from .material_models import (
    deviatoric,
    VoceIsotropicHardeningModel,
    NoIsotropicHardeningModel,
    ChabocheModelUniAxial,
    ChabocheModelMultiAxial,
    OhnoWangIIModelUniAxial,
    OhnoWangIIModelMultiAxial,
    AbdelKarimOhnoModelMultiAxial,
    KangModelMultiAxial,
)
from .solver import (
    UnifiedMaterialSolverUniAxial,
    UnifiedMaterialSolverMultiAxial,
)

__all__ = [
    "deviatoric",
    "VoceIsotropicHardeningModel",
    "NoIsotropicHardeningModel",
    "ChabocheModelUniAxial",
    "ChabocheModelMultiAxial",
    "OhnoWangIIModelUniAxial",
    "OhnoWangIIModelMultiAxial",
    "AbdelKarimOhnoModelMultiAxial",
    "KangModelMultiAxial",
    "UnifiedMaterialSolverUniAxial",
    "UnifiedMaterialSolverMultiAxial",
]
