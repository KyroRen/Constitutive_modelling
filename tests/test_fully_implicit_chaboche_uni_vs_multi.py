#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of fully-implicit uniaxial vs implicit multi-axial Chaboche model.

This script:
- reads calibrated Chaboche parameters from Steel_Chaboche.json,
- builds both uniaxial and multi-axial Chaboche material models,
- runs a ±1% strain history in xx direction for a chosen number of cycles,
- compares σ11 from uniaxial fully-implicit and multi-axial implicit (fully implicit in tensors),
- reports the wall-clock time used by each solver.

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
    ChabocheModelUniAxial,
    ChabocheModelMultiAxial,
)
from plasticity_solver.solver import (
    UnifiedMaterialSolverUniAxial,
    UnifiedMaterialSolverMultiAxial,
)


def load_chaboche_from_json(json_path: Path, nu_default: float = 0.3):
    """
    Load calibrated Chaboche parameters from a JSON file and build
    uniaxial and multi-axial material models.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file containing calibrated parameters.
    nu_default : float, optional
        Default Poisson's ratio to use if not present in the JSON.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'E' : Young's modulus (MPa)
        - 'sigma_y' : initial yield stress (MPa)
        - 'nu' : Poisson's ratio
        - 'model_uni' : ChabocheModelUniAxial instance
        - 'model_multi' : ChabocheModelMultiAxial instance
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mat_props = data.get("material_properties", {})
    params = data.get("calibrated_parameters", {})

    E = float(mat_props["E_MPa"])
    sigma_y = float(mat_props["sigy0_MPa"])

    # Isotropic hardening parameters (Voce)
    R_inf = float(params.get("Q_MPa", 0.0))
    b = float(params.get("b", 0.0))
    isotropic_model = VoceIsotropicHardeningModel(R_inf=R_inf, b=b)

    # Kinematic hardening parameters
    C = np.array(params["C_MPa"], dtype=float)
    gamma = np.array(params["gamma"], dtype=float)

    # Poisson's ratio is not stored in this JSON; use a reasonable default
    nu = float(data.get("material_properties", {}).get("nu", nu_default))

    model_uni = ChabocheModelUniAxial(isotropic_model=isotropic_model, C=C, gamma=gamma)
    model_multi = ChabocheModelMultiAxial(
        isotropic_model=isotropic_model, C=C, gamma=gamma, E=E, nu=nu
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
    Convenience function that builds uniaxial and multi-axial solvers
    from a Chaboche JSON calibration file.

    Returns
    -------
    tuple
        (solver_uni, solver_multi)
    """
    cfg = load_chaboche_from_json(json_path)

    E = cfg["E"]
    sigma_y = cfg["sigma_y"]
    nu = cfg["nu"]

    # Uniaxial solver: use fully_implicit mode
    solver_uni = UnifiedMaterialSolverUniAxial(
        E=E,
        yield_stress=sigma_y,
        material_model=cfg["model_uni"],
        method="fully_implicit",
        precision="high",
    )

    # Multi-axial solver: implicit mode (fully implicit in tensors)
    solver_multi = UnifiedMaterialSolverMultiAxial(
        E=E,
        nu=nu,
        yield_stress=sigma_y,
        material_model=cfg["model_multi"],
        method="fully_implicit",
        precision="high",
    )

    return solver_uni, solver_multi


def generate_uniaxial_strain_history_uniaxial_solver(
    solver_uni: UnifiedMaterialSolverUniAxial,
    amp_pos: float = 0.01,
    amp_neg: float = -0.01,
    n_cycles: int = 10,
    n_points_per_segment: int = 100,
):
    """
    Generate a cyclic uniaxial strain path using the existing helper
    in the uniaxial solver, for consistency with previous tests.

    Parameters
    ----------
    solver_uni : UnifiedMaterialSolverUniAxial
        Uniaxial solver instance (used only for the helper).
    amp_pos, amp_neg : float
        Maximum and minimum axial strain (e11).
    n_cycles : int
        Number of cycles.
    n_points_per_segment : int
        Number of points per loading/unloading segment.

    Returns
    -------
    np.ndarray
        1D array of total axial strain values (e11) over time.
    """
    path, _, _ = solver_uni.generate_cyclic_path(
        amp_pos=amp_pos,
        amp_neg=amp_neg,
        n_cycles=n_cycles,
        n_points=n_points_per_segment,
    )
    return path


def build_multiaxial_strain_history_from_e11(e11_history: np.ndarray) -> np.ndarray:
    """
    Build a 3x3 strain tensor history from a given e11 history.

    For the multi-axial solver's strain-controlled mode, we only need
    to provide an initial guess for the full tensor. The solver will
    internally adjust transverse strains to enforce a uniaxial stress
    state (σ22 = σ33 = 0).

    Parameters
    ----------
    e11_history : np.ndarray
        1D array of axial strain values (e11).

    Returns
    -------
    np.ndarray
        Array of shape (N, 3, 3) with only the [0,0] component filled.
    """
    n = len(e11_history)
    strain_tensors = np.zeros((n, 3, 3), dtype=float)
    strain_tensors[:, 0, 0] = e11_history
    return strain_tensors


def compare_uniaxial_vs_multiaxial(json_path: Path, n_cycles: int = 2):
    """
    Run a comparison between uniaxial fully-implicit and multi-axial implicit
    Chaboche solvers for ±1% axial strain over 10 cycles.
    """
    solver_uni, solver_multi = build_solvers_from_json(json_path)

    # Generate uniaxial strain history (e11)
    e11_history = generate_uniaxial_strain_history_uniaxial_solver(
        solver_uni=solver_uni,
        amp_pos=0.01,
        amp_neg=-0.01,
        n_cycles=n_cycles,
        n_points_per_segment=100,
    )

    # Run uniaxial fully-implicit simulation (measure time)
    t0 = time.perf_counter()
    sigma_uni, strain_uni = solver_uni.run_strain_controlled(e11_history)
    t1 = time.perf_counter()

    # Build 3x3 strain tensors for multi-axial solver
    strain_tensors = build_multiaxial_strain_history_from_e11(e11_history)

    # Run multi-axial implicit simulation (fully implicit in tensorial form, measure time)
    t2 = time.perf_counter()
    stresses_multi, strains_multi = solver_multi.run_strain_controlled(strain_tensors)
    t3 = time.perf_counter()

    # Extract σ11 from multi-axial result
    sigma_multi = stresses_multi[:, 0, 0]

    # Basic consistency checks
    if len(sigma_uni) != len(sigma_multi):
        raise RuntimeError(
            f"Length mismatch between uniaxial ({len(sigma_uni)}) and multi-axial "
            f"({len(sigma_multi)}) stress histories."
        )

    # Compute simple error metrics
    diff = sigma_multi - sigma_uni
    max_abs_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))

    print("=== Fully-implicit Chaboche: Uniaxial vs Multi-axial (sigma11) ===")
    print(f"Number of cycles         : {n_cycles}")
    print(f"Number of points         : {len(e11_history)}")
    print(f"Max |sigma11_multi - sigma11_uni|: {max_abs_diff:.6f} MPa")
    print(f"RMS  difference          : {rms_diff:.6f} MPa")
    print()
    print("Wall-clock times (seconds):")
    print(f"  Uniaxial solver        : {t1 - t0:.6f}")
    print(f"  Multi-axial solver     : {t3 - t2:.6f}")
    print()
    print("Sample values (index, e11, sigma11_uni, sigma11_multi, difference):")
    for idx in np.linspace(0, len(e11_history) - 1, num=5, dtype=int):
        print(
            f"{idx:6d}  "
            f"{e11_history[idx]: .6e}  "
            f"{sigma_uni[idx]: .6f}  "
            f"{sigma_multi[idx]: .6f}  "
            f"{diff[idx]: .6f}"
        )

    # Visualise hysteresis loops (σ11 vs e11)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        e11_history,
        sigma_uni,
        label="Uniaxial (fully-implicit)",
        linewidth=1.5,
    )
    ax.plot(
        e11_history,
        sigma_multi,
        "--",
        label="Multi-axial (implicit)",
        linewidth=1.5,
    )
    ax.set_xlabel("Strain e11 [-]")
    ax.set_ylabel("Stress sigma11 [MPa]")
    ax.set_title("Chaboche hysteresis loops: uniaxial vs multi-axial (fully-implicit)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    # Optional: show difference as a separate figure for diagnostics
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(e11_history, diff, color="tab:red", linewidth=1.2)
    ax2.set_xlabel("Strain e11 [-]")
    ax2.set_ylabel("Delta sigma11 [MPa] (multi - uni)")
    ax2.set_title("Difference between multi-axial and uniaxial sigma11")
    ax2.grid(True, linestyle=":", linewidth=0.5)
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Assume the JSON file is located next to this script.
    this_dir = Path(__file__).resolve().parent
    config_dir = this_dir.parent / "config"
    json_file = config_dir / "Steel_Chaboche.json"

    if not json_file.is_file():
        raise FileNotFoundError(f"Could not find Steel_Chaboche.json at: {json_file}")

    # Compare for a larger number of cycles to assess performance
    compare_uniaxial_vs_multiaxial(json_file, n_cycles= 5)

