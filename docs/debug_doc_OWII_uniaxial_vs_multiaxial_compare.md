# OWII Uniaxial vs Multiaxial: sqrt(3/2) and sqrt(2/3) Comparison

This document compares where the two formulations use similar or different scaling factors. Zig-zag patterns in multi-axial results often stem from 1D-3D factor inconsistencies.

**Chaboche reference**: Chaboche uniaxial vs multiaxial match (max |Δσ11| ~8e-7 MPa). Its conventions are used as the correct reference:
- dp = plastic multiplier (equivalent plastic strain increment)
- For uniaxial stress: d_ε_p_11 = dp (flow_direction has n_11 = sqrt(2/3), so sqrt(3/2)*dp*n_11 = dp)

## 1. Plastic Modulus / Consistency Denominator

| Location | Uniaxial | Multiaxial |
|----------|----------|------------|
| **OhnoWangIIModelUniAxial.plastic_multiplier** (line ~836) | `denominator = E + h_total` | — |
| **OhnoWangIIModelMultiAxial.plastic_multiplier** (line ~1131) | — | `denominator = 3 * self.G + h_total` |
| Relation | E | 3*G = 1.5*E/(1+ν) |

For ν=0.3: 3*G ≈ 1.15*E. **Different denominators** → different dp for same f and h.

---

## 2. Stress Update (Plastic Correction)

| Location | Uniaxial | Multiaxial |
|----------|----------|------------|
| **UnifiedMaterialSolverUniAxial** (line ~1111) | `stress = trial_stress - E * dp * sign` | — |
| **UnifiedMaterialSolverMultiAxial** (lines ~457, 616, 639) | — | `stress = trial_stress - 2*μ * d_ε_p` with `d_ε_p = sqrt(3/2) * dp * flow_direction` |

**Chaboche reference**: Multiaxial uses `flow_direction = s/||s||`; for uniaxial σ11, n_11 = sqrt(2/3) (deviatoric direction), so d_ε_p_11 = sqrt(3/2)*dp*n_11 = **dp**.
- Uniaxial: Δσ_11 = -E * dp
- Multiaxial: Δσ_11 = -2*μ * d_ε_p_11 = -2*μ * dp

For ν=0.3: 2*μ = E/(1+ν) ≈ 0.77*E. **Different stress drop per dp** (E vs 2μ).

---

## 3. Backstress Evolution – Hardening Term

| Location | Uniaxial | Multiaxial |
|----------|----------|------------|
| **update_backstress** | `delta_alpha = C * dp * sign - ...` | `delta_alpha = (2/3) * C * d_ε_p - ...` |
| **update_backstress_implicit** | `RHS = alpha_old + C * dp * sign` | `numerator = alpha_old + (2/3) * C * d_ε_p` |

**Chaboche reference**: For uniaxial stress, d_ε_p_11 = dp (n_11 = sqrt(2/3), so sqrt(3/2)*dp*n_11 = dp). Same convention in both ChabocheUniAxial and ChabocheMultiAxial.
- Multiaxial hardening in 11: (2/3)*C*d_ε_p_11 = (2/3)*C*dp
- Uniaxial: C*dp

**Uniaxial hardening is (3/2)× stronger** than multiaxial in α_11 for same dp.

---

## 4. Backstress Evolution – Recovery Term (pow_term)

| Location | Uniaxial | Multiaxial |
|----------|----------|------------|
| **pow_term** | `(ai_abs / denom)^m` with ai_abs = \|α\| | `(ai_bar / denom)^m` with ai_bar = sqrt(3/2)*\|\|α\|\| |

For uniaxial α_11 = α: ||α|| = |α|*sqrt(3/2), so ai_bar = (3/2)*|α|.
- Uniaxial: (|α|/(C/γ))^m
- Multiaxial: ((3/2)*|α|/(C/γ))^m = 1D_pow_term * (3/2)^m

**Multiaxial recovery is (3/2)^m stronger** than uniaxial (for m=10: ≈58×).

---

## 5. Plastic Strain Increment Definition

| Location | Uniaxial | Multiaxial |
|----------|----------|------------|
| **dε_p** | Implicit: magnitude dp, direction = sign | `d_ε_p = sqrt(3/2) * dp * flow_direction` |

**Chaboche reference**: Both use dp = plastic multiplier. For uniaxial stress, d_ε_p_11 = dp (n_11 = sqrt(2/3)).
Uniaxial: dε_p = dp (scalar, direction from sign).
Multiaxial: d_ε_p = sqrt(3/2)*dp*flow_direction → d_ε_p_11 = dp for uniaxial (n_11 = sqrt(2/3)).

---

## 6. Kinematic Hardening Modulus h_kin

| Location | Uniaxial | Multiaxial |
|----------|----------|------------|
| **compute_plastic_modulus** | `h_i = C_i - γ_i * pow_term * n * ai` (no sqrt(3/2) on C term) | `h_i = C_i - sqrt(3/2) * γ_i * pow_term * n_dot_ai` |

Uniaxial: n*ai = sign*α_i. Multiaxial: sqrt(3/2)*γ*(n:a_i). The sqrt(3/2) factor appears only in the multiaxial recovery part.

---

## Summary: Factor Differences

| Quantity | Uniaxial | Multiaxial | Ratio (multi/uni) |
|----------|----------|------------|-------------------|
| Plastic modulus denom | E | 3*G | ~1.15 |
| Stress drop per dp | E | 2*μ = E/(1+ν) | ~0.77 |
| Hardening increment | C*dp | (2/3)*C*dp | 2/3 |
| Recovery (pow base) | \|α\| | (3/2)*\|α\| | (3/2)^m |
| h_kin recovery term | γ*pow*n*α | sqrt(3/2)*γ*pow*(n:α) | — |

The zig-zag may be caused by:
1. **Different dp** from E vs 3*G in the consistency denominator.
2. **Different stress drop** from E vs 2μ = E/(1+ν).
3. **Transverse strain Newton solve** in run_strain_controlled (tol=1e-8, maxiter=20) — non-smooth convergence can cause oscillations.
4. **Sub-stepping**: multiaxial uses n_substeps=20 (line 406) — small steps can amplify numerical noise.

---

## Code Reference (exact locations)

| Item | File | Line(s) |
|------|------|---------|
| Uniaxial stress update | Module_Solver.py | 1111 |
| Multiaxial stress update | Module_Solver.py | 457, 616, 639 |
| Uniaxial plastic_multiplier (E+h) | Module_material_models.py | 836 |
| Multiaxial plastic_multiplier (3*G+h) | Module_material_models.py | 1131-1134 |
| Multiaxial d_epsilon_p = sqrt(3/2)*dp*n | Module_Solver.py | 457, 525, 550, 616, 640 |
| OWII UniAxial pow_term \|α\| | Module_material_models.py | 823-825 |
| OWII MultiAxial pow_term ai_bar | Module_material_models.py | 1112-1117 |
| OWII MultiAxial hardening (2/3)*C | Module_material_models.py | 1156 |
| Transverse strain Newton (zig-zag source?) | Module_Solver.py | 799-819 |
| root() tol for coupled residual | Module_Solver.py | 633 |

---

## Potential Zig-zag Mitigations

1. **Tighten transverse Newton**: Reduce `tol` from 1e-8 to 1e-10 in line 819.
2. **Tighten coupled root solve**: Reduce `tol` from 1e-8 in line 633.
3. **Align plastic modulus denominator**: For uniaxial-stress loading, consider using a denominator consistent with 1D (requires detecting uniaxial case).
4. **Sub-stepping**: Multiaxial explicit uses n_substeps=20 (line 405); fully_implicit uses adaptive. Try finer substeps if zig-zag persists.
