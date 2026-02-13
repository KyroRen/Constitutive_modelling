#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-axial Chaboche and OWII test under prescribed stress history.

This script:
- reads calibrated Chaboche parameters from Steel_Chaboche.json,
- reads calibrated OWII parameters from Steel_OWII_m_10_10_10_10.json,
- builds multi-axial Chaboche and OWII material models and solvers (fully_implicit),
- applies the same stress-controlled loading path to both:
      s11: [0, 400, 0] MPa
      s31: [0, 700, 0] MPa
  repeated for a given number of cycles,
- visualises:
      s11 vs e11,
      s31 vs e31,
      von Mises effective stress vs effective plastic strain p.

IMPORTANT:
- This is a standalone test script and does not modify any module files.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cross-platform compatibility (CI/headless)
import matplotlib.pyplot as plt

from plasticity_solver.material_models import (
    VoceIsotropicHardeningModel,
    NoIsotropicHardeningModel,
    ChabocheModelMultiAxial,
    OhnoWangIIModelMultiAxial,
    deviatoric,
)
from plasticity_solver.solver import UnifiedMaterialSolverMultiAxial


def load_multiaxial_chaboche_from_json(json_path: Path, nu_default: float = 0.3):
    """
    Load calibrated Chaboche parameters from a JSON file and build
    a multi-axial material model along with basic elastic data.

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

    model_multi = ChabocheModelMultiAxial(
        isotropic_model=isotropic_model, C=C, gamma=gamma, E=E, nu=nu
    )

    return {
        "E": E,
        "sigma_y": sigma_y,
        "nu": nu,
        "model_multi": model_multi,
    }


def build_multiaxial_solver_from_json(json_path: Path) -> UnifiedMaterialSolverMultiAxial:
    """
    Build a multi-axial Chaboche solver from a JSON calibration file.
    """
    cfg = load_multiaxial_chaboche_from_json(json_path)
    E = cfg["E"]
    sigma_y = cfg["sigma_y"]
    nu = cfg["nu"]

    solver = UnifiedMaterialSolverMultiAxial(
        E=E,
        nu=nu,
        yield_stress=sigma_y,
        material_model=cfg["model_multi"],
        method="fully_implicit",
        precision="high",
    )
    return solver


def load_multiaxial_owii_from_json(json_path: Path, nu_default: float = 0.3):
    """
    Load calibrated Ohno-Wang II parameters from a JSON file and build
    a multi-axial OWII material model along with basic elastic data.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file (e.g. Steel_OWII_m_10_10_10_10.json).
    nu_default : float, optional
        Default Poisson's ratio to use if not present in the JSON.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'E' : Young's modulus (MPa)
        - 'sigma_y' : initial yield stress (MPa)
        - 'nu' : Poisson's ratio
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

    model_multi = OhnoWangIIModelMultiAxial(
        isotropic_model=isotropic_model, C=C, gamma=gamma, m=m, E=E, nu=nu
    )

    return {
        "E": E,
        "sigma_y": sigma_y,
        "nu": nu,
        "model_multi": model_multi,
    }


def build_multiaxial_owii_solver_from_json(json_path: Path) -> UnifiedMaterialSolverMultiAxial:
    """
    Build a multi-axial OWII solver from a JSON calibration file (fully_implicit).
    """
    cfg = load_multiaxial_owii_from_json(json_path)
    E = cfg["E"]
    sigma_y = cfg["sigma_y"]
    nu = cfg["nu"]

    solver = UnifiedMaterialSolverMultiAxial(
        E=E,
        nu=nu,
        yield_stress=sigma_y,
        material_model=cfg["model_multi"],
        method="fully_implicit",
        precision="high",
    )
    return solver


def generate_stress_history_s11_s31(
    s11_points, s31_points, n_cycles: int = 10, n_points_per_segment: int = 100
) -> np.ndarray:
    """
    Generate a multi-axial stress tensor history from prescribed s11 and s31 control points.

    The path is piecewise linear between the control points, and repeated for n_cycles.

    Parameters
    ----------
    s11_points : list or array-like
        Control-point values for σ11 (MPa), e.g. [0, 400, 200, 0].
    s31_points : list or array-like
        Control-point values for σ31 (MPa), e.g. [0, 400, 700, 0].
    n_cycles : int
        Number of cycles to repeat the basic 4-point path.
    n_points_per_segment : int
        Number of points per segment between consecutive control points.

    Returns
    -------
    np.ndarray
        Stress history, shape (N, 3, 3), where N = n_cycles * (len(points)-1) * n_points_per_segment.
    """
    s11_points = np.asarray(s11_points, dtype=float)
    s31_points = np.asarray(s31_points, dtype=float)

    if s11_points.shape != s31_points.shape:
        raise ValueError("s11_points and s31_points must have the same shape.")
    if s11_points.ndim != 1 or s11_points.size < 2:
        raise ValueError("At least two control points are required for the stress path.")

    base_s11 = []
    base_s31 = []

    # Build one cycle as piecewise linear segments between control points
    for i in range(len(s11_points) - 1):
        seg_s11 = np.linspace(s11_points[i], s11_points[i + 1], n_points_per_segment)
        seg_s31 = np.linspace(s31_points[i], s31_points[i + 1], n_points_per_segment)
        base_s11.append(seg_s11)
        base_s31.append(seg_s31)

    base_s11 = np.concatenate(base_s11)
    base_s31 = np.concatenate(base_s31)

    # Repeat for n_cycles
    s11_history = np.tile(base_s11, n_cycles)
    s31_history = np.tile(base_s31, n_cycles)

    n_total = s11_history.size
    stress_tensors = np.zeros((n_total, 3, 3), dtype=float)

    # Apply σ11 and σ31 (and enforce symmetry σ31 = σ13)
    stress_tensors[:, 0, 0] = s11_history
    stress_tensors[:, 2, 0] = s31_history
    stress_tensors[:, 0, 2] = s31_history

    return stress_tensors


def compute_von_mises(stress_tensors: np.ndarray) -> np.ndarray:
    """
    Compute von Mises effective stress for a history of stress tensors.

    Parameters
    ----------
    stress_tensors : np.ndarray
        Array of shape (N, 3, 3).

    Returns
    -------
    np.ndarray
        1D array of von Mises equivalent stress values (MPa).
    """
    n = stress_tensors.shape[0]
    sigma_eq = np.zeros(n, dtype=float)
    for i in range(n):
        s = deviatoric(stress_tensors[i])
        J2 = 0.5 * np.sum(s * s)
        sigma_eq[i] = np.sqrt(3.0 * J2)
    return sigma_eq


def run_stress_controlled_path(
    solver: UnifiedMaterialSolverMultiAxial,
    stress_history: np.ndarray,
):
    """
    Run a stress-controlled simulation for a given multi-axial stress history.

    This wrapper uses solver.compute_strain() step by step so that we can
    record the evolution of effective plastic strain p.

    Parameters
    ----------
    solver : UnifiedMaterialSolverMultiAxial
        Multi-axial solver instance (Chaboche).
    stress_history : np.ndarray
        Array of shape (N, 3, 3) with target stress tensors.

    Returns
    -------
    tuple
        (strain_history, stress_history_out, p_history)
        - strain_history      : np.ndarray, shape (N, 3, 3)
        - stress_history_out  : np.ndarray, shape (N, 3, 3)
        - p_history           : np.ndarray, shape (N,), accumulated equivalent plastic strain.
    """
    strains = []
    stresses = []
    p_hist = []

    # Ensure we start from a clean state
    solver.reset_state()

    for target_stress in stress_history:
        try:
            strain = solver.compute_strain(target_stress)
        except RuntimeError as exc:
            print(f"Stress-controlled step failed: {exc}")
            break

        strains.append(strain.copy())
        stresses.append(solver.current_stress.copy())
        p_hist.append(solver.p)

    return np.array(strains), np.array(stresses), np.array(p_hist)


def plot_results(
    strain_history: np.ndarray,
    stress_history: np.ndarray,
    p_history: np.ndarray,
    strain_owii=None,
    stress_owii=None,
    p_owii=None,
):
    """
    Plot:
    - s11 vs e11,
    - s31 vs e31,
    - von Mises effective stress vs effective plastic strain p.

    If OWII results are provided (strain_owii, stress_owii, p_owii), both Chaboche
    and OWII are plotted on the same axes.
    """
    e11 = strain_history[:, 0, 0]
    e31 = strain_history[:, 2, 0]
    s11 = stress_history[:, 0, 0]
    s31 = stress_history[:, 2, 0]
    sigma_eq = compute_von_mises(stress_history)

    # Figure 1: s11 vs e11
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(e11, s11, linewidth=1.5, label="Chaboche")
    if strain_owii is not None and stress_owii is not None:
        e11_o = strain_owii[:, 0, 0]
        s11_o = stress_owii[:, 0, 0]
        ax1.plot(e11_o, s11_o, "--", linewidth=1.5, label="OWII")
    ax1.set_xlabel("Strain e11 [-]")
    ax1.set_ylabel("Stress sigma11 [MPa]")
    ax1.set_title("sigma11 vs e11 (multi-axial, fully-implicit)")
    ax1.legend()
    ax1.grid(True, linestyle=":", linewidth=0.5)
    fig1.tight_layout()

    # Figure 2: s31 vs e31
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(e31, s31, linewidth=1.5, label="Chaboche")
    if strain_owii is not None and stress_owii is not None:
        e31_o = strain_owii[:, 2, 0]
        s31_o = stress_owii[:, 2, 0]
        ax2.plot(e31_o, s31_o, "--", linewidth=1.5, label="OWII")
    ax2.set_xlabel("Strain e31 [-]")
    ax2.set_ylabel("Stress sigma31 [MPa]")
    ax2.set_title("sigma31 vs e31 (multi-axial, fully-implicit)")
    ax2.legend()
    ax2.grid(True, linestyle=":", linewidth=0.5)
    fig2.tight_layout()

    # Figure 3: von Mises vs p
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(p_history, sigma_eq, linewidth=1.5, label="Chaboche")
    if stress_owii is not None and p_owii is not None:
        sigma_eq_o = compute_von_mises(stress_owii)
        ax3.plot(p_owii, sigma_eq_o, "--", linewidth=1.5, label="OWII")
    ax3.set_xlabel("Effective plastic strain p [-]")
    ax3.set_ylabel("Von Mises effective stress [MPa]")
    ax3.set_title("Effective stress vs effective plastic strain p")
    ax3.legend()
    ax3.grid(True, linestyle=":", linewidth=0.5)
    fig3.tight_layout()

    # Figure 4: strain path (e11 vs e31) and stress path (s11 vs s31)
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 4))
    ax4a.plot(e11, e31, linewidth=1.5, label="Chaboche")
    if strain_owii is not None:
        e11_o = strain_owii[:, 0, 0]
        e31_o = strain_owii[:, 2, 0]
        ax4a.plot(e11_o, e31_o, "--", linewidth=1.5, label="OWII")
    ax4a.set_xlabel("Strain e11 [-]")
    ax4a.set_ylabel("Strain e31 [-]")
    ax4a.set_title("Strain path: e11 vs e31")
    ax4a.legend()
    ax4a.grid(True, linestyle=":", linewidth=0.5)
    ax4a.set_aspect("equal", adjustable="datalim")

    ax4b.plot(s11, s31, linewidth=1.5, label="Chaboche")
    if stress_owii is not None:
        s11_o = stress_owii[:, 0, 0]
        s31_o = stress_owii[:, 2, 0]
        ax4b.plot(s11_o, s31_o, "--", linewidth=1.5, label="OWII")
    ax4b.set_xlabel("Stress sigma11 [MPa]")
    ax4b.set_ylabel("Stress sigma31 [MPa]")
    ax4b.set_title("Stress path: sigma11 vs sigma31")
    ax4b.legend()
    ax4b.grid(True, linestyle=":", linewidth=0.5)
    ax4b.set_aspect("equal", adjustable="datalim")

    fig4.tight_layout()

    plt.show()


def main():
    """
    Entry point for the multi-axial stress-controlled Chaboche and OWII test.
    """
    this_dir = Path(__file__).resolve().parent
    config_dir = this_dir.parent / "config"
    json_chaboche = config_dir / "Steel_Chaboche.json"
    json_owii = config_dir / "Steel_OWII_m_10_10_10_10.json"

    if not json_chaboche.is_file():
        raise FileNotFoundError(f"Could not find Steel_Chaboche.json at: {json_chaboche}")
    if not json_owii.is_file():
        raise FileNotFoundError(f"Could not find Steel_OWII_m_10_10_10_10.json at: {json_owii}")

    solver_chaboche = build_multiaxial_solver_from_json(json_chaboche)
    solver_owii = build_multiaxial_owii_solver_from_json(json_owii)

    # User-specified control points (MPa) - same load path for both models
    s11_points = [0.0, 400.0, 200, 0.0]
    s31_points = [0.0, 300.0, 500, 0.0]

    n_cycles = 10
    n_points_per_segment = 100

    stress_history = generate_stress_history_s11_s31(
        s11_points=s11_points,
        s31_points=s31_points,
        n_cycles=n_cycles,
        n_points_per_segment=n_points_per_segment,
    )

    print("Running Chaboche (fully-implicit)...")
    strains_chaboche, stresses_chaboche, p_chaboche = run_stress_controlled_path(
        solver=solver_chaboche,
        stress_history=stress_history,
    )

    print("Running OWII (fully-implicit)...")
    strains_owii, stresses_owii, p_owii = run_stress_controlled_path(
        solver=solver_owii,
        stress_history=stress_history,
    )

    print(f"Completed {len(stresses_chaboche)} stress-controlled steps "
          f"for {n_cycles} cycles (nominal) for both Chaboche and OWII.")

    plot_results(
        strain_history=strains_chaboche,
        stress_history=stresses_chaboche,
        p_history=p_chaboche,
        strain_owii=strains_owii,
        stress_owii=stresses_owii,
        p_owii=p_owii,
    )


if __name__ == "__main__":
    main()

