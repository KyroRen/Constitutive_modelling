# Fix: Multi-axial OWII Fully-Implicit Zig-Zag Problem

## Problem Description

The multi-axial Ohno-Wang II (OWII) fully-implicit solver produced **zig-zag (oscillatory) patterns** in the stress-strain response, while the multi-axial Chaboche model and the uniaxial OWII model were both smooth and stable. Before the fix, comparing the same uniaxial loading path:

| Metric | Before Fix | After Fix |
|---|---|---|
| Max \|sigma11_multi - sigma11_uni\| | **13.159 MPa** | **0.000003 MPa** |
| RMS difference | **7.769 MPa** | **0.000001 MPa** |

Two bugs were identified and fixed in `Module_Solver.py`. Both affected the `UnifiedMaterialSolverMultiAxial` class.

---

## Fix 1: Truly Implicit Backward Euler for OWII Backstress

**File:** `Module_Solver.py`, method `_compute_stress_internal_fully_implicit`, function `residual_coupled`

### Root Cause

The Ohno-Wang II backstress evolution law is:

    da_i = (2/3)*C_i*de_p - gamma_i * (a_bar_i / r_i)^m * dp * a_i

where `a_bar_i = sqrt(3/2 * a_i : a_i)` and `r_i = C_i / gamma_i`.

The backward Euler (fully implicit) discretisation should be:

    a_i_new * (1 + gamma_i * (a_bar_i_new / r_i)^m * dp) = a_i_old + (2/3)*C_i*de_p

Note: the `pow_term = (a_bar / r)^m` must be evaluated at **a_i_new** (the unknown) for a truly implicit scheme.

**The bug:** The multi-axial `OhnoWangIIModelMultiAxial.update_backstress_implicit()` evaluated the pow_term at **a_i_old** instead of **a_i_new**:

```python
# BEFORE (semi-implicit — pow_term at alpha_OLD):
ai_norm = np.linalg.norm(ai_old)          # <-- ai_old, NOT ai_new!
ai_bar = np.sqrt(3.0 / 2.0) * ai_norm
pow_term = (ai_bar / denom) ** self.m[i]   # <-- pow_term frozen at old state
new_backstress[i] = (ai_old + (2/3)*C*de_p) / (1 + gamma * pow_term * dp)
```

This made it a **semi-implicit** (IMEX) scheme disguised as fully-implicit. The `residual_coupled` function then set `res_alpha = alpha_new - update_backstress_implicit(alpha_old, dp)`, which trivially determined alpha_new from the formula — the root solver's alpha unknowns were not genuinely coupled.

In contrast, the **uniaxial** `OhnoWangIIModelUniAxial.update_backstress_implicit()` correctly solves a nonlinear scalar equation for `|alpha_new|` using `brentq`, with the pow_term evaluated at `alpha_new`.

### Why Chaboche Was Unaffected

For Chaboche, the backstress evolution is linear:

    da_i = (2/3)*C_i*de_p - gamma_i * dp * a_i

The backward Euler gives `a_new = (a_old + (2/3)*C*de_p) / (1 + gamma*dp)`. There is **no pow_term**, so using alpha_old vs alpha_new makes no difference — the equation is linear in a_new either way.

### Why This Caused Zig-Zag in OWII

For OWII with m=10, the pow_term `(a_bar/r)^10` amplifies any difference between a_old and a_new by a factor of up to 10x (derivative). When alpha is near the saturation limit (a_bar ~ r), pow_term ~ 1; when far from saturation, pow_term ~ 0. Using the wrong evaluation point causes:

- **Over-correction** when alpha moves toward saturation (pow_term at a_old is smaller than at a_new)
- **Under-correction** when alpha moves away from saturation

This creates alternating over/under-shoot at each strain increment — the zig-zag pattern.

### The Fix

Replaced the `residual_coupled` function to express the backward Euler equation directly as a residual, with the pow_term evaluated at `alpha_new` (the solver unknowns):

```python
# AFTER (truly implicit — pow_term at alpha_NEW):
def residual_coupled(x):
    dp_guess = x[0]
    alpha_new_list = [x[1+i*9 : 1+(i+1)*9].reshape(3,3) for i in ...]

    for i in range(n_components):
        ai_new = alpha_new_list[i]   # <-- from solver unknowns
        ai_old = self.backstress[i]

        recall_coeff = gamma[i]
        if has_pow_term:  # True for OWII
            ai_new_norm = np.linalg.norm(ai_new)
            ai_new_bar = sqrt(3/2) * ai_new_norm
            pow_val = (ai_new_bar / r_i) ** m[i]    # <-- evaluated at ai_NEW!
            recall_coeff *= pow_val

        # Truly implicit backward Euler residual
        res = ai_new * (1 + recall_coeff * dp) - ai_old - (2/3)*C[i]*de_p
```

This makes the coupled root-finding system genuinely nonlinear in alpha_new, and the `scipy.optimize.root` solver finds the self-consistent solution where `pow_term(alpha_new)` and `alpha_new` are mutually consistent.

The model-level `update_backstress_implicit()` in `Module_material_models.py` was **not changed** — it is still used by the IMEX method where the semi-implicit treatment is intentional.

---

## Fix 2: Yield-Detection State Not Restored in Transverse Newton

**File:** `Module_Solver.py`, method `run_strain_controlled`

### Root Cause

The `_compute_stress_fully_implicit` wrapper (line 297) uses `self._last_stress_tensor` to detect elastic-to-plastic transitions. When crossing the yield surface for the first time, it inserts an intermediate computation at the exact yield point before proceeding to the target strain. This two-step process ensures accuracy at the elastic-plastic transition.

In `run_strain_controlled`, a transverse Newton-Raphson loop iterates to find `e_t` (transverse strain) such that `sigma_22 = 0`. Each Newton iteration:
1. Restores the saved state
2. Calls `compute_stress` with a trial strain tensor
3. Returns sigma_22 as the residual

**The bug:** The `saved_state` dictionary saved/restored plastic_strain, backstress, current_stress, total_strain, R, p, and history lengths — but **not** `_last_stress_tensor` or `_last_f_value`. This meant:

- Newton iteration 1: `_last_stress_tensor` comes from the previous load step (correct)
- Newton iteration 2: `_last_stress_tensor` is now the result from iteration 1 (wrong — should be from previous load step)

This caused the yield-detection logic to behave inconsistently across Newton iterations: some iterations triggered the two-step yield crossing, others did not. For Chaboche (smooth, linear backstress), this inconsistency was negligible. For OWII (highly nonlinear pow_term), it contributed to the zig-zag.

### The Fix

Added `_last_stress_tensor` and `_last_f_value` to the saved state, and restored them at the start of each Newton iteration and before the final computation:

```python
saved_state = {
    # ... existing state variables ...
    '_last_stress_tensor': self._last_stress_tensor.copy() if hasattr(...) else None,
    '_last_f_value': getattr(self, '_last_f_value', None),
}

def transverse_residual(e_t):
    # Restore state (including yield-detection variables)
    # ... existing restores ...
    if saved_state['_last_stress_tensor'] is not None:
        self._last_stress_tensor = saved_state['_last_stress_tensor'].copy()
    self._last_f_value = saved_state['_last_f_value']
```

---

## Verification Results

### OWII (50 cycles, +/-1% strain, MHH_OWII_m_10_10_10_10.json)

```
Max |sigma11_multi - sigma11_uni|: 0.000003 MPa   (was 13.159105 MPa)
RMS difference                   : 0.000001 MPa   (was  7.768835 MPa)
```

### Chaboche (verified by user — no regression)

The Chaboche multi-axial solver continues to work correctly. The change is mathematically equivalent for Chaboche (no pow_term), so the residual differs only by a positive scalar factor `(1 + gamma*dp)` which does not change the root.

---

## Summary

| Fix | Location | What Changed | Impact |
|---|---|---|---|
| Truly implicit pow_term | `residual_coupled` in solver | pow_term evaluated at alpha_new instead of alpha_old | Eliminated zig-zag, reduced 1D-3D error by 6 orders of magnitude |
| Yield-detection state restore | `run_strain_controlled` | `_last_stress_tensor` saved/restored in transverse Newton | Consistent yield-point detection across Newton iterations |
