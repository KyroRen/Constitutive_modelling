# Uniaxial vs Multiaxial OWII Fully-Implicit: Key Differences

## 1. Solve Dimension

| Uniaxial | Multiaxial |
|----------|------------|
| **1 unknown**: dp only | **1 + 9×n unknowns**: dp + all α tensors |
| residual(dp) → f_new (scalar) | residual_coupled([dp, α₁...αₙ]) → [res_f, res_α...] |
| newton() or brentq() | root(..., method='hybr') |

## 2. Strain-Controlled Path

| Uniaxial | Multiaxial |
|----------|------------|
| Direct: e11 history → compute_stress(e11) | **Transverse Newton**: find e_t s.t. σ22=0 for each e11 |
| No inner Newton | Each transverse_residual(e_t) calls compute_stress → triggers root() |
| 1 solve per strain point | Many root() calls per strain point (Newton iterations) |

## 3. Stress Update

| Uniaxial | Multiaxial |
|----------|------------|
| σ = trial - E·dp·sign | σ = trial - 2μ·√(3/2)·dp·flow_direction |

## 4. Backstress Update

| Uniaxial | Multiaxial |
|----------|------------|
| update_backstress_implicit(α, dp, **sign**) | update_backstress_implicit(α, dp, **flow_direction**) |

## 5. Sub-stepping

| Uniaxial | Multiaxial |
|----------|------------|
| n_substeps when |Δε| > 0.001 (up to 10) | n_substeps = 1 (default from run_strain_controlled) |

## 6. Root Cause of Zig-zag

Multiaxial uses a **coupled (1+9n)-dim root** while uniaxial uses a **1-dim root**. The coupled system is over-parameterized: α is uniquely determined by dp via update_backstress_implicit. We can reduce multiaxial to a **dp-only solve** (like uniaxial) for consistency and stability.
