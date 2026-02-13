#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of fully-implicit uniaxial vs multi-axial Ohno-Wang II model.

This script:
- loads Ohno-Wang II parameters from Steel_OWII_m_10_10_10_10.json,
- builds both uniaxial and multi-axial OWII material models,
- runs a +/-1% strain history in xx direction for a chosen number of cycles,
- compares sigma11 from uniaxial and multi-axial fully-implicit solvers,
- reports error metrics and displays wall-clock times on the figures.

IMPORTANT:
- This is a standalone test script and does not modify any module files.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cross-platform compatibility (CI/headless)
import matplotlib.pyplot as plt

from plasticity_solver.material_models import (
    VoceIsotropicHardeningModel,
    NoIsotropicHardeningModel,
    OhnoWangIIModelUniAxial,
    OhnoWangIIModelMultiAxial,
)
from plasticity_solver.solver import (
    UnifiedMaterialSolverUniAxial,
    UnifiedMaterialSolverMultiAxial,
)


def load_owii_from_json(json_path: Path, nu_default: float = 0.3):
    """
    Load Ohno-Wang II parameters from a JSON calibration file and build
    uniaxial and multi-axial material models.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file (e.g. Steel_OWII_m_10_10_10_10.json).
    nu_default : float, optional
        Default Poisson's ratio if not present in the JSON.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'E' : Young's modulus (MPa)
        - 'sigma_y' : initial yield stress (MPa)
        - 'nu' : Poisson's ratio
        - 'model_uni' : OhnoWangIIModelUniAxial instance
        - 'model_multi' : OhnoWangIIModelMultiAxial instance
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mat_props = data.get("material_properties", {})
    params = data.get("calibrated_parameters", {})
    cal_info = data.get("calibration_info", {})

    E = float(mat_props["E_MPa"])
    sigma_y = float(mat_props["sigy0_MPa"])

    # OWII kinematic hardening: C, gamma, m
    C = np.array(params["C_MPa"], dtype=float)
    gamma = np.array(params["gamma"], dtype=float)
    m = np.array(cal_info.get("m_values", [10.0] * len(C)), dtype=float)
    if len(m) != len(C):
        m = np.full(len(C), 10.0, dtype=float)

    # Isotropic hardening (Voce)
    R_inf = float(params.get("Q_MPa", 0.0))
    b = float(params.get("b", 0.0))
    if R_inf != 0 or b != 0:
        isotropic_model = VoceIsotropicHardeningModel(R_inf=R_inf, b=b)
    else:
        isotropic_model = NoIsotropicHardeningModel()

    nu = float(mat_props.get("nu", nu_default))

    model_uni = OhnoWangIIModelUniAxial(
        isotropic_model=isotropic_model, C=C, gamma=gamma, m=m
    )
    model_multi = OhnoWangIIModelMultiAxial(
        isotropic_model=isotropic_model, C=C, gamma=gamma, m=m, E=E, nu=nu
    )

    return {
        "E": E,
        "sigma_y": sigma_y,
        "nu": nu,
        "model_uni": model_uni,
        "model_multi": model_multi,
    }


def build_solvers_from_json(json_path: Path):
    """
    Build uniaxial and multi-axial OWII solvers from a JSON calibration file.

    Returns
    -------
    tuple
        (solver_uni, solver_multi)
    """
    cfg = load_owii_from_json(json_path)
    E = cfg["E"]
    sigma_y = cfg["sigma_y"]
    nu = cfg["nu"]

    solver_uni = UnifiedMaterialSolverUniAxial(
        E=E,
        yield_stress=sigma_y,
        material_model=cfg["model_uni"],
        method="fully_implicit",
        precision="high",
    )
    solver_multi = UnifiedMaterialSolverMultiAxial(
        E=E,
        nu=nu,
        yield_stress=sigma_y,
        material_model=cfg["model_multi"],
        method="fully_implicit",
        precision="high",
    )
    return solver_uni, solver_multi


def generate_uniaxial_strain_history(
    solver_uni: UnifiedMaterialSolverUniAxial,
    amp_pos: float = 0.01,
    amp_neg: float = -0.01,
    n_cycles: int = 50,
    n_points_per_segment: int = 100,
) -> np.ndarray:
    """Generate a cyclic uniaxial strain path (e11) using the solver helper."""
    path, _, _ = solver_uni.generate_cyclic_path(
        amp_pos=amp_pos,
        amp_neg=amp_neg,
        n_cycles=n_cycles,
        n_points=n_points_per_segment,
    )
    return path


def build_multiaxial_strain_history_from_e11(e11_history: np.ndarray) -> np.ndarray:
    """Build 3x3 strain tensor history from e11 history."""
    n = len(e11_history)
    strain_tensors = np.zeros((n, 3, 3), dtype=float)
    strain_tensors[:, 0, 0] = e11_history
    return strain_tensors


def compare_uniaxial_vs_multiaxial_owii(
    json_path: Path, n_cycles: int = 50
):
    """
    Run a comparison between uniaxial and multi-axial fully-implicit OWII
    solvers for +/-1% axial strain over n_cycles.
    """
    solver_uni, solver_multi = build_solvers_from_json(json_path)

    e11_history = generate_uniaxial_strain_history(
        solver_uni=solver_uni,
        amp_pos=0.01,
        amp_neg=-0.01,
        n_cycles=n_cycles,
        n_points_per_segment=100,
    )

    # Run uniaxial simulation (measure time)
    t0 = time.perf_counter()
    sigma_uni, strain_uni = solver_uni.run_strain_controlled(e11_history)
    t1 = time.perf_counter()
    time_uni = t1 - t0

    strain_tensors = build_multiaxial_strain_history_from_e11(e11_history)

    # Run multi-axial simulation (measure time)
    t2 = time.perf_counter()
    stresses_multi, strains_multi = solver_multi.run_strain_controlled(strain_tensors)
    t3 = time.perf_counter()
    time_multi = t3 - t2

    sigma_multi = stresses_multi[:, 0, 0]

    if len(sigma_uni) != len(sigma_multi):
        raise RuntimeError(
            f"Length mismatch: uniaxial ({len(sigma_uni)}) vs multi-axial ({len(sigma_multi)})"
        )

    diff = sigma_multi - sigma_uni
    max_abs_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))

    print("=== Fully-implicit OWII: Uniaxial vs Multi-axial (sigma11) ===")
    print(f"JSON file               : {json_path.name}")
    print(f"Number of cycles        : {n_cycles}")
    print(f"Number of points        : {len(e11_history)}")
    print(f"Max |sigma11_multi - sigma11_uni|: {max_abs_diff:.6f} MPa")
    print(f"RMS difference          : {rms_diff:.6f} MPa")
    print()
    print("Wall-clock times (seconds):")
    print(f"  Uniaxial solver       : {time_uni:.6f}")
    print(f"  Multi-axial solver    : {time_multi:.6f}")
    print()
    print("Sample values (index, e11, sigma11_uni, sigma11_multi, diff):")
    for idx in np.linspace(0, len(e11_history) - 1, num=5, dtype=int):
        print(
            f"  {idx:6d}  {e11_history[idx]: .6e}  "
            f"{sigma_uni[idx]: .6f}  {sigma_multi[idx]: .6f}  {diff[idx]: .6f}"
        )

    # Figure 1: Hysteresis loops with time in title
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        e11_history,
        sigma_uni,
        label="Uniaxial OWII (fully-implicit)",
        linewidth=1.5,
    )
    ax.plot(
        e11_history,
        sigma_multi,
        "--",
        label="Multi-axial OWII (fully-implicit)",
        linewidth=1.5,
    )
    ax.set_xlabel("Strain e11 [-]")
    ax.set_ylabel("Stress sigma11 [MPa]")
    ax.set_title(
        f"OWII hysteresis loops: uniaxial vs multi-axial ({n_cycles} cycles)\n"
        f"Uniaxial: {time_uni:.3f} s  |  Multi-axial: {time_multi:.3f} s"
    )
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    # Figure 2: Difference with time annotation
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.plot(e11_history, diff, color="tab:red", linewidth=1.2)
    ax2.set_xlabel("Strain e11 [-]")
    ax2.set_ylabel("Delta sigma11 [MPa] (multi - uni)")
    ax2.set_title(
        f"Difference between multi-axial and uniaxial sigma11\n"
        f"Uniaxial: {time_uni:.3f} s  |  Multi-axial: {time_multi:.3f} s  |  "
        f"Max |diff|: {max_abs_diff:.4f} MPa"
    )
    ax2.grid(True, linestyle=":", linewidth=0.5)
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    this_dir = Path(__file__).resolve().parent
    config_dir = this_dir.parent / "config"
    json_file = config_dir / "Steel_OWII_m_10_10_10_10.json"

    if not json_file.is_file():
        raise FileNotFoundError(f"Could not find {json_file.name} at: {json_file}")

    compare_uniaxial_vs_multiaxial_owii(json_file, n_cycles=50)
