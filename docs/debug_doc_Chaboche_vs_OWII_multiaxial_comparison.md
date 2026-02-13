# Multi-axial Chaboche vs OWII: Solution Differences and Zig-zag Analysis

Comparison of why Chaboche multi-axial gives smooth hysteresis while OWII multi-axial shows zig-zag patterns.

---

## 1. Shared Solver Infrastructure (Identical)

Both models use the same `UnifiedMaterialSolverMultiAxial` with `method="fully_implicit"`:

| Component | Implementation | Location |
|-----------|----------------|----------|
| `run_strain_controlled` | Transverse Newton: find e_t s.t. σ22=0 | Module_Solver.py 781-839 |
| `_compute_stress_internal_fully_implicit` | Coupled residual: [dp, α₁...αₙ] | Module_Solver.py 566-665 |
| `root()` | scipy.optimize.root, method='hybr', tol=1e-8 | Module_Solver.py 633 |
| Transverse Newton | scipy.optimize.newton, tol=1e-8, maxiter=20 | Module_Solver.py 819 |
| flow_direction | s / \|\|s\|\| | 595 |
| d_epsilon_p | sqrt(3/2) * dp * flow_direction | 616, 640 |
| Stress update | trial - 2μ * d_epsilon_p | 616, 644 |
| Initial dp guess | f / (E + h_plastic) | 625-626 |

**Solver path is identical** — no model-specific branches. The zig-zag must come from the material model behaviour inside the residual.

---

## 2. Material Model Differences

### 2.1 update_backstress_implicit

| Aspect | Chaboche | OWII |
|--------|----------|------|
| **Formula** | α_new = (α_old + (2/3)C·dε_p) / (1 + γ·dp) | α_new = (α_old + (2/3)C·dε_p) / (1 + γ·**pow_term**·dp) |
| **Denominator** | 1 + γ·dp (linear in dp) | 1 + γ·(ā/(C/γ))^m·dp |
| **Nonlinearity** | Linear in α | **Highly nonlinear**: pow_term = (ai_bar/denom)^m |
| **Smoothness** | Smooth | Steep when m large (e.g. m=10) |

### 2.2 compute_plastic_modulus (used for initial guess)

| Chaboche | OWII |
|----------|------|
| h_i = C_i - sqrt(3/2)·γ_i·(n:α_i) | h_i = C_i - sqrt(3/2)·γ_i·**pow_term**·(n:α_i) |
| Linear in α | pow_term = (ai_bar/denom)^m — very sensitive near threshold |

### 2.3 Key Numerical Difference: pow_term

OWII recovery uses **pow_term = (ai_bar / (C/γ))^m**. For m=10:

- ai_bar = 0.9*(C/γ) → pow_term ≈ 0.35
- ai_bar = 1.0*(C/γ) → pow_term = 1.0
- ai_bar = 1.1*(C/γ) → pow_term ≈ 2.6

Small changes in ai_bar near the saturation threshold produce large changes in pow_term, hence in:

- Plastic modulus h (initial dp guess)
- Implicit backstress denominator

This makes the residual **much more nonlinear** for OWII than for Chaboche.

---

## 3. Likely Zig-zag Causes

### 3.1 Nonlinear Residual

- **Chaboche**: Residual is relatively smooth; root() converges stably.
- **OWII**: pow_term^(m) with m≈10 creates steep gradients. root(..., tol=1e-8) may:
  - Converge to slightly different dp/α at neighbouring strain points
  - Produce small but visible oscillations in σ11

### 3.2 Initial Guess Sensitivity

- Initial guess: `dp0 = f / (E + h_plastic)`.
- OWII: h_plastic depends strongly on pow_term. Small variations in backstress → large changes in h_plastic → different dp0.
- Poor or inconsistent initial guess can lead to different convergence paths and oscillatory σ11.

### 3.3 Transverse Newton Interaction

- Transverse Newton finds e_t so that σ22 = 0.
- Each residual evaluation calls `compute_stress`, which runs the coupled root().
- For OWII, root() can converge to slightly different solutions for similar (e11, e_t).
- The effective mapping e_t ↔ σ11 becomes less smooth, contributing to zig-zag.

### 3.4 Semi-implicit pow_term

- OWII uses **α_old** in pow_term (not α_new).
- The implicit solve is closed-form but approximates the fully implicit case.
- This can introduce small inconsistencies that accumulate or oscillate over steps.

---

## 4. Summary: Why Chaboche Smooth, OWII Zig-zag

| Factor | Chaboche | OWII |
|--------|----------|------|
| Recovery term | Linear γ·α·dp | Nonlinear (ā/r)^m·dp |
| Residual curvature | Mild | Steep (m=10) |
| Initial guess stability | Stable | Sensitive to h_plastic |
| root() convergence | Robust | More sensitive to tol/guess |
| Numerical noise | Small impact | Amplified by pow_term |

---

## 5. Suggested Mitigations for OWII Zig-zag

1. **Tighten root tolerance**: Try `tol=1e-10` or `tol=1e-12` in line 633.
2. **Tighten transverse Newton**: Try `tol=1e-10` in line 819.
3. **Improve initial guess**: Use previous step dp as part of x0 when strain increment is small.
4. **Sub-stepping**: Call compute_stress with `n_substeps > 1` for larger strain steps to reduce increment size per root() call.
5. **Fully implicit pow_term** (research): Use α_new in pow_term; requires iterative scheme instead of closed form.
6. **Smoothing**: Apply light smoothing to σ11 output for visualization (does not fix underlying oscillation).

---

## 6. Code Reference

| Item | File | Line |
|------|------|------|
| Coupled root (dp, α) | Module_Solver.py | 633 |
| Transverse Newton | Module_Solver.py | 819 |
| Initial dp guess | Module_Solver.py | 625-626 |
| Chaboche update_backstress_implicit | Module_material_models.py | 1035-1044 |
| OWII update_backstress_implicit | Module_material_models.py | 1148-1164 |
| OWII pow_term | Module_material_models.py | 1159 |
