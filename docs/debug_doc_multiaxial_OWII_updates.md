## Multi-axial Ohno–Wang II model: current state and test setup

This note documents how the **multi-axial OWII model** is implemented and how it
is used in the fully-implicit multi-axial solver, plus the dedicated comparison
test script that was added.

### 1. Multi-axial OWII material model

**Location:** `Module_material_models.py`  
**Class:** `OhnoWangIIModelMultiAxial`

Key methods:

- `compute_plastic_modulus(self, backstress, state_vars)`  
  Computes the total plastic modulus for multi-axial OWII:

  \[
  h = \sum_i \left[ C_i
        - \sqrt{\frac{3}{2}}\,\gamma_i
          \left(\frac{\bar{a}_i}{C_i/\gamma_i}\right)^{m_i}
          (n : a_i)
      \right]
      + h_\mathrm{iso}
  \]

  where \(\bar{a}_i = \sqrt{\frac{3}{2}\,a_i:a_i} = \sqrt{3/2}\,\lVert a_i\rVert\) is the Mises-equivalent of the deviatoric backstress tensor (J2 convention; \(\lVert\cdot\rVert\) is Frobenius norm).

- `yield_function(self, trial_stress, total_backstress, sigma_y)`  
  Standard J2 (von Mises) yield:

  \[
  f = \sqrt{3 J_2} - \sigma_y, \quad
  J_2 = \tfrac{1}{2} s:s, \quad s = \text{dev}(\sigma - \alpha)
  \]

- `update_backstress(self, backstress, d_epsilon, flow_direction, state_vars)`  
  **Explicit** OWII update:

  \[
  d\alpha_i =
    \frac{2}{3} C_i\,d\varepsilon_p
    - \gamma_i \left(\frac{\bar{a}_i}{C_i/\gamma_i}\right)^{m_i}
      dp \, a_i,
  \quad d\varepsilon_p = \sqrt{\tfrac{3}{2}}\,dp\,n,
  \quad \bar{a}_i = \sqrt{\tfrac{3}{2}\,a_i:a_i}
  \]

- `update_backstress_implicit(self, backstress_tensors_old, dp, flow_direction)`  
  **Implicit (backward Euler) OWII update**:

  \[
  a_i^{n+1} =
    \frac{a_i^n + \frac{2}{3} C_i\,d\varepsilon_p}
         {1 + \gamma_i \left(\frac{\bar{a}_i^n}{C_i/\gamma_i}\right)^{m_i} dp},
  \quad \bar{a}_i^n = \sqrt{\tfrac{3}{2}\,a_i^n:a_i^n}
  \]

  This provides a closed-form implicit update for each backstress tensor
  component.

- `update_state_vars_implicit(self, R_old, p_old, dp)`  
  Calls `isotropic_model.update_implicit(...)` for consistent implicit
  evolution of isotropic hardening variables `(R, p)`.

Together, these methods provide a **fully-implicit incremental update** for the
multi-axial OWII backstress and isotropic hardening.

### 2. Multi-axial solver method options

**Location:** `Module_Solver.py`  
**Class:** `UnifiedMaterialSolverMultiAxial`

The multi-axial solver supports three integration methods:

| Method              | Description |
|---------------------|-------------|
| `explicit`          | Full explicit: plastic multiplier from plastic modulus, explicit backstress and state updates |
| `implicit_explicit` | IMEX: implicit yield consistency (solve for dp), explicit backstress and isotropic hardening updates |
| `fully_implicit`    | Fully-implicit: coupled residual over dp and all backstress tensors |

Use `solver.set_method('fully_implicit')` for fully-implicit (default) or  
`solver.set_method('implicit_explicit')` for the IMEX scheme.

### 3. Coupling with the multi-axial fully-implicit solver

**Method:** `_compute_stress_internal_implicit`

Inside `_compute_stress_internal_implicit`, the solver constructs a coupled
residual over the plastic multiplier `dp` and all backstress tensor components:

```python
def residual_coupled(x):
    dp_guess = x[0]
    alpha_new_list = [...]

    # 1) Backstress evolution residual using the material's implicit update
    alpha_old_list = self.backstress
    alpha_updated_list = self.material_model.update_backstress_implicit(
        alpha_old_list, dp_guess, flow_direction
    )
    # residuals_alpha = alpha_new_list - alpha_updated_list

    # 2) Yield condition residual at the updated state
    total_backstress_new = np.sum(alpha_new_list, axis=0)
    R_new, _ = self.material_model.update_state_vars_implicit(
        self.R, self.p, dp_guess
    )
    d_epsilon_p = np.sqrt(3./2.) * dp_guess * flow_direction
    stress_new = trial_stress_sub - 2 * self.mu * d_epsilon_p
    res_f, _ = self.material_model.yield_function(
        stress_new, total_backstress_new, self.sigma_y + R_new
    )
    ...
```

The root-finder (`scipy.optimize.root`) solves this system for `dp` and all
backstress components simultaneously. When the `material_model` is an instance
of `OhnoWangIIModelMultiAxial`, this machinery uses:

- `OhnoWangIIModelMultiAxial.update_backstress_implicit(...)` for kinematic
  hardening (OWII law, backward Euler); and
- `OhnoWangIIModelMultiAxial.update_state_vars_implicit(...)` for isotropic
  hardening.

Therefore, **the multi-axial OWII model is already integrated in a fully-implicit
fashion** when `UnifiedMaterialSolverMultiAxial(method="fully_implicit")` is used.

### 4. Dedicated comparison script: Uni-axial vs Multi-axial OWII

**New file:** `test_fully_implicit_owii_uni_vs_multi.py`

Purpose:

- build **1D OWII + `UnifiedMaterialSolverUniAxial(method="fully_implicit")`**,
- build **3D OWII + `UnifiedMaterialSolverMultiAxial(method="fully_implicit")`**,
- apply a **±1% e11** cyclic strain history for a configurable number of cycles,
- compare σ11–e11 hysteresis loops and report the difference between
  uniaxial and multi-axial responses, as well as CPU times.

Key steps:

1. **Build models and solvers**

   - Uses `NoIsotropicHardeningModel` (pure kinematic hardening) and a local
     parameter set for `C`, `gamma`, `m`.
   - Uniaxial OWII: `OhnoWangIIModelUniAxial`
   - Multi-axial OWII: `OhnoWangIIModelMultiAxial`

2. **Generate strain history**

   - Uses `UnifiedMaterialSolverUniAxial.generate_cyclic_path(...)` to create
     a 1D e11(t) path (0 → +0.01 → −0.01 → +0.01 etc.).
   - Embeds e11(t) into 3×3 tensors with only `ε11` non-zero for the
     multi-axial solver.

3. **Run simulations**

   - Uniaxial: `solver_uni.run_strain_controlled(e11_history)`
   - Multi-axial: `solver_multi.run_strain_controlled(strain_tensors)`

4. **Post-processing**

   - Extract σ11 from both runs; compute max and RMS differences.
   - Measure wall-clock times for each solver.
   - Plot:
     - σ11 vs e11 for uniaxial and multi-axial OWII on the same figure;
     - Δσ11 vs e11 as a separate diagnostic plot.

Usage:

```bash
cd d:\OneDrive\PostDoc\ELABORATE\005_Solver\Step_5
python test_fully_implicit_owii_uni_vs_multi.py
```

This script does **not** modify any module files; it only imports and exercises
the existing implementations of the uniaxial and multi-axial OWII models and
solvers.

