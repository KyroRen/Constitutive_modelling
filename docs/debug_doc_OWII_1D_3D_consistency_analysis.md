# OWII Uniaxial vs Multiaxial Consistency Analysis

## Observed Behaviour

- **Chaboche**: Max |σ11_multi - σ11_uni| ≈ 8e-7 MPa (excellent match)
- **OWII (before fix)**: Max |σ11_multi - σ11_uni| ≈ 86 MPa (large discrepancy)
- **OWII (after ai_bar fix)**: Max |σ11_multi - σ11_uni| ≈ 13 MPa (improved, residual gap)

Both use the same fully-implicit solvers and uniaxial-stress strain-controlled setup.

## Potential Sources of OWII 1D-3D Inconsistency

### 1. Backstress norm in the recovery term

**1D OWII** uses `|α|` (scalar) in:
```
pow_term = (|α| / (C/γ))^m
```

**3D OWII** should use `ā = sqrt(3/2 * α:α)` (Mises-equivalent backstress) in:
```
pow_term = (ā / (C/γ))^m,   where ā = sqrt(3/2 * α:α) = sqrt(3/2) * ||α||
```

**Implementation (fixed)**: Now uses `ai_bar = sqrt(3/2) * ||α||` in the pow_term.

For uniaxial backstress α_11 = α, α_22 = α_33 = -α/2:
- `||α|| = |α_11| * sqrt(3/2)`, so `ā = sqrt(3/2) * ||α|| = (3/2) * |α_11|`
- 1D uses |α|; 3D (correct form) would use ā = (3/2)|α_11| in pow_term
- If 3D incorrectly uses ||α|| = |α_11|*sqrt(3/2) instead of ā, the pow_term is too small by factor (sqrt(3/2))^m → weaker recovery in 3D than intended

### 2. Hardening term scaling

**1D**: hardening increment = C * dp
**3D**: hardening = (2/3) * C * dε_p, with dε_p_11 = dp for uniaxial
- So 3D α_11 hardening = (2/3) * C * dp
- 3D has (2/3)× the 1D hardening rate
- Weaker hardening → lower backstress → lower σ (expect σ_multi < σ_uni)
- Again contradicts σ_multi > σ_uni

### 3. Plastic modulus and elastic stiffness

- **1D**: denominator = E + h, stress drop = E * dp
- **3D**: denominator = 3*G + h, stress drop = 2*μ * dε_p
- For uniaxial: 2*μ = E/(1+ν), so 3D stress drop per unit dp differs from 1D by factor (1+ν)
- This can affect the balance between elastic and plastic response

### 4. Implicit formulation detail

- **1D OWII implicit**: solves nonlinear equation for α_new using |α_new| in the residual
- **3D OWII implicit**: uses closed form with pow_term evaluated at **α_old** (not α_new)
- The 3D version is thus a "semi-implicit" approximation for the recovery term

## Fix Attempts (all reverted)

1. **Norm (wrong direction)**: Used `ai_eff = ||α|| / sqrt(3/2)` in 3D
   - Result: Max diff increased from ~86 to ~185 MPa. Reverted.
   - Note: The correct fix may be the opposite — use `ā = sqrt(3/2) * ||α||` so that 3D uses the Mises-equivalent backstress. **Not yet tried.**

2. **Hardening**: Use C instead of (2/3)*C in 3D hardening
   - Result: Max diff increased to ~1025 MPa. Reverted.

3. **Use ā = sqrt(3/2)*||α|| in 3D pow_term**: Implemented. Max diff reduced from ~86 MPa to ~13 MPa.

## Suggested Next Steps

1. **Review literature**: Confirm the standard Ohno–Wang II multiaxial form (norm definition, 1D reduction, and scaling conventions).
2. **Consult plasticity expert**: The interaction between hardening scaling, norm definition, and elastic/plastic coupling may require domain expertise to resolve.
3. **Consider calibration source**: If OWII parameters were calibrated from 1D tests, the 3D formulation may need to be derived by proper reduction from the 1D form rather than by applying the standard tensorial form directly.

## Code Review: Hardcoded Shortcuts

Checked for hardcoded values or shortcuts in OWII and Chaboche models:

- **No problematic hardcodes found.** Both OWII and Chaboche use parameters from the JSON/config.
- **Numerical tolerances** (e.g. 1e-16, 1e-12) are used only for division-by-zero safety.
- **Scaling factors** (sqrt(3/2), 2/3) follow standard J2 plasticity conventions.

## Code Locations

- **OhnoWangIIModelUniAxial**: `Module_material_models.py` ~line 795
- **OhnoWangIIModelMultiAxial**: `Module_material_models.py` ~line 1084
- Chaboche 1D and 3D share the same C, gamma and achieve agreement; OWII uses the same parameters but shows ~86 MPa discrepancy.
