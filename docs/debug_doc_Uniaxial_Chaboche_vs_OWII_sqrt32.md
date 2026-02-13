# Uniaxial Chaboche vs Uniaxial OWII: sqrt(3/2) Comparison

Comparison of where sqrt(3/2) (from J2 tensor operations) appears or is absent in the two uniaxial models.

## J2 Origin of sqrt(3/2)

In J2 plasticity, tensor operations introduce:
- Equivalent stress: σ_eq = sqrt(3/2 * s:s)
- Equivalent plastic strain: dp = sqrt(2/3) * ||dε_p||
- Mises-equivalent of deviatoric tensor α: ᾱ = sqrt(3/2) * ||α||
- For uniaxial α_11 only: ||α|| = sqrt(3/2)*|α_11|, so ᾱ = (3/2)*|α_11|

---

## Side-by-Side: Uniaxial Chaboche vs Uniaxial OWII

| Item | Chaboche Uniaxial | OWII Uniaxial | sqrt(3/2) present? |
|------|-------------------|---------------|--------------------|
| **dp** | dp = d_epsilon (plastic multiplier) | dp = d_epsilon | No (same in both) |
| **Stress update** | σ = trial - E*dp*sign | σ = trial - E*dp*sign | No |
| **Backstress hardening** | C * dp * sign | C * dp * sign | No |
| **Backstress recovery** | γ * α * dp | γ * (\|α\|/denom)^m * α * dp | No |
| **pow_term (recovery)** | N/A (linear) | (ai_abs/denom)^m with ai_abs = \|α\| | **No** — uses \|α\|, not ᾱ |
| **h_kin (plastic modulus)** | C - γ*sign*α | C - γ*pow_term*n*α | No |
| **Isotropic p update** | p += dp | p += dp | No |

---

## Key Finding: No sqrt(3/2) in Either Uniaxial Model

Both uniaxial Chaboche and uniaxial OWII use **no sqrt(3/2)** anywhere. They treat:
- dp as the equivalent plastic strain increment (no scaling)
- α as the 1D scalar backstress (no ᾱ = sqrt(3/2)*||α||)

---

## Where sqrt(3/2) Would Appear for 1D–3D Consistency

If we map the 3D J2 formulation to uniaxial:

| Quantity | 3D (multiaxial) | 1D mapping | Uniaxial Chaboche | Uniaxial OWII |
|----------|-----------------|------------|-------------------|---------------|
| Backstress measure in recovery | ai_bar = sqrt(3/2)*\|\|α\|\| | ᾱ = (3/2)*\|α_11\| | — | Uses \|α\| only |
| Hardening coefficient | (2/3)*C | (2/3)*C for dα_11 | Uses C | Uses C |

**OWII recovery**: 3D uses (ai_bar/denom)^m with ai_bar = sqrt(3/2)*||α||. For uniaxial, ai_bar = (3/2)*|α|. Uniaxial OWII uses (|α|/denom)^m — missing the (3/2) factor.

**Chaboche**: No pow_term, so no ai_bar. Both 1D and 3D use γ*α*dp for recovery.

---

## Code References

| Model | File | Lines |
|-------|------|-------|
| Chaboche update_backstress | Module_material_models.py | 569–574 |
| Chaboche update_backstress_implicit | Module_material_models.py | 576–602 |
| Chaboche compute_plastic_modulus | Module_material_models.py | 540–554 |
| OWII update_backstress | Module_material_models.py | 842–850 |
| OWII update_backstress_implicit | Module_material_models.py | 853–917 |
| OWII compute_plastic_modulus | Module_material_models.py | 807–828 |
| Uniaxial stress update | Module_Solver.py | 1111 |

---

## Conclusion

- **Chaboche uniaxial**: No sqrt(3/2); linear recovery γ*α*dp.
- **OWII uniaxial**: No sqrt(3/2); pow_term uses |α| instead of ᾱ = (3/2)*|α|.
- For 1D–3D consistency, OWII uniaxial recovery could use ai_bar = sqrt(3/2)*|α| or (3/2)*|α| in the pow_term, matching the 3D convention.
