# Module Verification & Documentation: Solver

**File**: `Module_Solver.py`  
**Classes**: `UnifiedMaterialSolverUniAxial`, `UnifiedMaterialSolverMultiAxial`  
**Cross-Platform**: Compatible with Windows, macOS, and Linux

This document details the architecture, algorithms, and implementation of the unified material solvers, mapping the mathematical theory directly to the Python implementation.

---

## 1. Architecture Overview

The `UnifiedMaterialSolverUniAxial` and `UnifiedMaterialSolverMultiAxial` act as numerical integrators (Drivers) for the constitutive laws.

| Component | Responsibility | Code Attribute |
| :--- | :--- | :--- |
| **Global Driver** | Controls strain ref, sub-stepping | `compute_stress(self, total_strain)` |
| **Material** | Defines $\Phi$ (Yield), $H$ (Hardening) | `self.material_model` |
| **Integrator** | Solves $\Delta p$ for $\Phi_{n+1}=0$ | `_solve_plastic_multiplier_...` |
| **Method Control** | Switches between explicit/implicit | `set_method(method, precision)` |

### 1.1 Available Methods

Both solvers support multiple integration methods:

- **`explicit`**: Forward Euler integration (faster, less stable for large steps)
- **`implicit`**: Semi-implicit integration (more stable, recommended for most cases)
- **`fully_implicit`**: Fully implicit integration (most stable, for `UnifiedMaterialSolverUniAxial` only)

### 1.2 Precision Levels

Three precision levels are available:

- **`standard`**: Default tolerance `1e-8` (recommended for most applications)
- **`high`**: Tolerance `1e-9` (for higher accuracy requirements)
- **`scientific`**: Tolerance `1e-10` (for research-grade precision)

**Usage Example**:
```python
solver = UnifiedMaterialSolverUniAxial(E, yield_stress, model, method='implicit', precision='high')
solver.set_method('fully_implicit', precision='scientific')  # Switch method and precision
```

---

## 2. Fully Implicit Solver Strategy

**Method**: `_solve_plastic_multiplier_fully_implicit`

The goal is to find the scalar plastic multiplier $\Delta p$ that satisfies the yield criterion at the **end** of the step.

### 2.1 The Global Residual Equation

**Theory**:
We seek $\Delta p \ge 0$ such that:
$$ f(\Delta p) = \sigma_{trial} - E \Delta p \cdot \text{sign} - \sum \alpha_{i, n+1}(\Delta p) - \sigma_y - R_{n+1}(\Delta p) = 0 $$

**Implementation**:
```python
def residual(dp):
    # 1. Enforce non-negative constraint
    if dp < 0: return 1e8 * abs(dp)
    
    dp_reg = max(dp, 1e-16)

    # 2. Implicit Update of Internal Variables
    # The KEY difference: We calculate alpha_{n+1} and R_{n+1} 
    # fully consistently for *this specific candidate dp*
    if hasattr(self.material_model, 'update_backstress_implicit'):
        new_backstress = self.material_model.update_backstress_implicit(
            self.backstress, dp_reg, sign
        )

    new_R, _ = self.material_model.isotropic_model.update_implicit(
        R_old, getattr(self, 'p', 0.0), dp_reg
    )

    # 3. Evaluate Yield Function at n+1 state
    stress_new = trial_stress - self.E * dp_reg * sign
    total_backstress_new = np.sum(new_backstress)
    
    f_new, _ = self.material_model.yield_function(
        stress_new, total_backstress_new, self.sigma_y + new_R
    )
    
    return f_new
```

### 2.2 Robust Bracket Search (V-Shape Detection)

**Problem**: The residual $f(\Delta p)$ involves absolute values $|\sigma - \alpha|$, creating a "V-shape". If the search step doubles too aggressively ($x_{k+1} = 2 x_k$), it can jump across the V-pit entirely, seeing only positive values.

**Theory**:
If $f(x_{k+1}) > f(x_k)$ while $f(x) > 0$, we have leaped past the minimum. The root lies in $(x_k, x_{k+1}]$.

**Implementation**:
```python
prev_val = val_0
prev_upper = 0.0

for _ in range(50):
    val = residual(upper)
    
    # 1. Found a negative point -> Valid Bracket
    if val <= 0:
        bracket_found = True
        break
        
    # 2. V-Shape Detection: Value started increasing!
    # We missed the root in the previous interval.
    if val > prev_val:
        # Refined Grid Search in [prev_upper, upper]
        sub_steps = np.linspace(prev_upper, upper, 20)
        for sub_dp in sub_steps:
             if residual(sub_dp) <= 0:
                 upper = sub_dp
                 bracket_found = True
                 break
                 
    prev_val = val
    prev_upper = upper
    upper *= 2.0  # Expansion step
```

---

## 3. Sub-stepping Logic

**Method**: `_compute_stress_implicit`, `_compute_stress_fully_implicit`

To ensure convergence for sparse data, we divide large strain steps.

### 3.1 Implicit Method Sub-stepping

**Theory**:
Split $\Delta \epsilon$ into $N$ steps of size $\delta \epsilon \approx 0.05\%$.
$$ N = \max(1, \lceil \frac{|\Delta \epsilon|}{0.0005} \rceil) $$

**Implementation**:
```python
strain_increment_size = abs(d_strain)

# Target 0.05% per step for implicit method
if strain_increment_size > 0.0005:
    n_substeps = max(int(strain_increment_size / 0.0005), 4)
else:
    n_substeps = 1

# Safety Cap: 5000 (Allows up to 100% strain step)
n_substeps = min(n_substeps, 5000)
```

### 3.2 Fully Implicit Method Sub-stepping

**Theory**:
For fully implicit method, larger sub-steps are acceptable:
$$ N = \max(1, \lceil \frac{|\Delta \epsilon|}{0.001} \rceil) $$

**Implementation**:
```python
strain_increment_size = abs(d_strain)

# Target 0.1% per step for fully implicit (more stable)
if strain_increment_size > 0.001:
    n_substeps = max(int(strain_increment_size / 0.001), 4)
else:
    n_substeps = 1

# Safety Cap: 10 (more efficient for fully implicit)
n_substeps = min(n_substeps, 10)
```

---

## 4. Stress-Controlled Simulation

**Method**: `compute_strain(self, target_stress)`

For stress-controlled loading, the solver uses a Newton-Raphson method to find the strain that produces the target stress.

### 4.1 Unified Convergence Strategy

The solver uses a multi-method approach with adaptive tolerance levels:

```python
# Multiple solver methods tried in sequence
methods = ['hybr', 'lm', 'broyden1']
tolerance_levels = [tol, tol*10, tol*100]

# Adaptive initial guess based on stress magnitude
if stress_magnitude > 100:  # High stress
    scaling_factor = 0.3
elif stress_magnitude > 10:  # Medium stress
    scaling_factor = 0.5
else:  # Low stress
    scaling_factor = 0.7
```

### 4.2 Enhanced Fallback Strategy

If the primary solver fails, the code uses progressively smaller initial guesses and multiple fallback strategies to ensure convergence.

---

## 5. Cross-Platform Compatibility

**Status**: ✅ Fully compatible with Windows, macOS, and Linux

### 5.1 Platform Support

- **Windows**: Tested on Windows 11, handles path separators and encoding correctly
- **macOS**: Compatible with Unix path conventions
- **Linux**: Fully supported with standard Python libraries

### 5.2 Dependencies

All dependencies are cross-platform:
- `numpy`: Numerical computations
- `scipy`: Optimization routines (`newton`, `root`, `brentq`)
- `matplotlib`: Plotting (optional, for diagnostics)

### 5.3 File Operations

The solver module itself does not perform file I/O operations. All file operations are handled by `Module_material_models.py` with cross-platform path support.

---

## 6. Usage Examples

### 6.1 Basic Strain-Controlled Simulation

```python
from Module_material_models import ChabocheModelUniAxial, NoIsotropicHardeningModel
from Module_Solver import UnifiedMaterialSolverUniAxial

# Create material model
isotropic = NoIsotropicHardeningModel()
C = [100000.0, 50000.0, 10000.0]
gamma = [1000.0, 100.0, 10.0]
model = ChabocheModelUniAxial(isotropic_model=isotropic, C=C, gamma=gamma)

# Create solver
E = 200000.0
yield_stress = 300.0
solver = UnifiedMaterialSolverUniAxial(E, yield_stress, model, method='implicit')

# Run simulation
strain_history = np.linspace(0, 0.01, 100)
stresses, strains = solver.run_strain_controlled(strain_history)
```

### 6.2 Stress-Controlled Simulation

```python
# Create solver (same as above)
solver = UnifiedMaterialSolverUniAxial(E, yield_stress, model, method='implicit')

# Run stress-controlled simulation
stress_history = np.linspace(0, 400, 100)
strains, stresses = solver.run_stress_controlled(stress_history)
```

### 6.3 Method Switching During Simulation

```python
# Start with explicit method
solver = UnifiedMaterialSolverUniAxial(E, yield_stress, model, method='explicit')

# Switch to implicit for better stability
solver.set_method('implicit', precision='high')

# Get current method info
info = solver.get_method_info()
print(f"Method: {info['method']}, Precision: {info['precision']}")
```

---

## 7. Diagnostic Tools

### 7.1 Stress Control Diagnostics

```python
# Diagnose convergence issues in stress control
solver.diagnose_stress_control(target_stress)

# Print convergence summary
solver.print_convergence_summary()
```

### 7.2 Method Information

```python
# Get detailed method information
info = solver.get_method_info()
# Returns: {'method': 'implicit', 'precision': 'high', 'yield_tolerance': 1e-9, ...}
```
