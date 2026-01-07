# Module Verification & Documentation: Material Models

**File**: `Module_material_models.py`  
**Classes**: Multiple material models (Chaboche, Ohno-Wang II, Abdel-Karim & Ohno, Kang, etc.)  
**Cross-Platform**: Compatible with Windows, macOS, and Linux

This document maps the constitutive theory of material models to their robust numerical implementation, with a focus on the Ohno-Wang II model.

---

## 1. Implicit Backstress Update

**Method**: `update_backstress_implicit`

The core challenge is solving the highly non-linear differential equation for backstress $\alpha$ over a finite step $\Delta p$.

### 1.1 The Discrete Equation (Backward Euler)

**Theory**:
$$ \alpha_{n+1} = \alpha_n + \underbrace{\frac{2}{3} C \Delta \epsilon^p}_{C \Delta p \cdot \text{sign}} - \gamma \Delta p \left( \frac{|\alpha_{n+1}|}{C/\gamma} \right)^m \alpha_{n+1} $$

Rearranging to solve for $\alpha_{n+1}$:
$$ \alpha_{n+1} \left[ 1 + \gamma \Delta p \left( \frac{|\alpha_{n+1}|}{C/\gamma} \right)^m \right] = \underbrace{\alpha_n + C \Delta p \cdot \text{sign}}_{\text{Trial Backstress (RHS)}} $$

Let $y = |\alpha_{n+1}|$ be the magnitude. Taking the absolute value of both sides:
$$ y \left[ 1 + \gamma \Delta p \left( \frac{y}{threshold} \right)^m \right] = |\text{RHS}| $$

**Implementation**:
```python
# 1. Setup Constants
threshold = C / gam
RHS = alpha_old + C * dp * sign
RHS_mag = abs(RHS)

# 2. Define Residual Function for Magnitude y
def residual_mag(y):
     # y * (1 + gamma * dp * (y/thresh)^m) - |RHS| = 0
     term_pow = (y / threshold) ** m
     return y * (1.0 + gam * dp * term_pow) - RHS_mag
```

### 1.2 Robust Solver (Brentq)

**Why Brentq?**: 
The function $f(y) = y(1 + \dots y^m) - C$ is strictly monotonic increasing.
*   $f(0) = -C < 0$
*   $f(C) > 0$ (since term in brackets > 1)
Thus, a unique root exists in $[0, |RHS|]$. Brentq is guaranteed to find it.

**Implementation**:
```python
try:
    from scipy.optimize import brentq
    
    # 3. Solve for y (Magnitude)
    # Bounds: [0, RHS_mag] is mathematically guaranteed to bracket the root
    y_sol = brentq(residual_mag, 0.0, RHS_mag, xtol=1e-12, maxiter=50)
    
    # 4. Recover sign
    # Since alpha_n+1 has same sign as Trial RHS (Equation structure)
    RHS_sign = np.sign(RHS) if abs(RHS) > 0 else 1.0
    x = y_sol * RHS_sign

except Exception:
    # 5. Fallback Bisection (if brentq fails internally)
    # ... manual bisection loop ...
```

---

## 2. Parameter Requirements

For this solver to be physically meaningful, the parameters must retain physical consistency.

| Parameter | Symbol | Code Variable | Constraint |
| :--- | :--- | :--- | :--- |
| **Modulus** | $C$ | `const[0]` (C) | $> 0$ |
| **Recovery** | $\gamma$ | `const[1]` (gamma) | $> 0$ |
| **Exponent** | $m$ | `const[2]` (m) | $\ge 0$ |

**Note on $m=0$**:
If $m=0$, the term $(y/threshold)^m = 1$. The equation becomes linear:
$$ \alpha_{n+1} [ 1 + \gamma \Delta p ] = \text{RHS} $$
$$ \alpha_{n+1} = \frac{\text{RHS}}{1 + \gamma \Delta p} $$
The code handles this:
```python
if m == 0:
    new_backstress[i] = RHS / (1.0 + gam * dp)
    continue
```

---

## 3. Material Model Loading

**Function**: `load_and_create_model(kinematic_file, isotropic_file=None, model_type='uniaxial')`

### 3.1 Cross-Platform File Loading

The material parameter loading function supports multiple path formats and ensures cross-platform compatibility:

**Supported Path Formats**:
- String paths: `"materials/chaboche_params.txt"`
- `pathlib.Path` objects: `Path("materials/chaboche_params.txt")`
- Absolute and relative paths

**Implementation**:
```python
from pathlib import Path

# All of these work:
params1 = parse_material_params("materials/params.txt")  # String path
params2 = parse_material_params(Path("materials/params.txt"))  # Path object
params3 = parse_material_params("/absolute/path/to/params.txt")  # Absolute path
```

### 3.2 UTF-8 Encoding

All file operations use explicit UTF-8 encoding for cross-platform compatibility:

```python
def parse_material_params(file_path):
    file_path = Path(file_path)  # Convert to Path for compatibility
    with open(file_path, 'r', encoding='utf-8', newline=None) as f:
        # Handles Windows (\r\n), Unix (\n), and Mac (\r) line endings
        for line in f:
            # Parse parameters...
```

### 3.3 Parameter File Format

Material parameter files use a simple `key = value` format:

```
E = 200000.0
nu = 0.3
sigy = 300.0
c = [100000.0, 50000.0, 10000.0]
gamma = [1000.0, 100.0, 10.0]
m = [2.0, 2.0, 2.0]
```

**Notes**:
- Parameter names are case-sensitive
- Arrays can be specified as Python lists
- Comments (lines starting with `#`) are ignored
- Empty lines are ignored

---

## 4. Available Material Models

### 4.1 Uniaxial Models

| Model Class | Description | Key Parameters |
| :--- | :--- | :--- |
| `ChabocheModelUniAxial` | Standard Chaboche kinematic hardening | `C`, `gamma` |
| `ChabocheTModelUniAxial` | Chaboche with threshold (NLK-T) | `C`, `gamma`, `al` |
| `OhnoWangIIModelUniAxial` | Ohno-Wang II kinematic hardening | `C`, `gamma`, `m` |

### 4.2 Multiaxial Models

| Model Class | Description | Key Parameters |
| :--- | :--- | :--- |
| `ChabocheModelMultiAxial` | 3D Chaboche model | `C`, `gamma`, `E`, `nu` |
| `OhnoWangIIModelMultiAxial` | 3D Ohno-Wang II model | `C`, `gamma`, `m`, `E`, `nu` |
| `AbdelKarimOhnoModelMultiAxial` | Abdel-Karim & Ohno model | `C`, `gamma`, `mu`, `E`, `nu` |
| `KangModelMultiAxial` | Kang model | `C`, `gamma`, `mu`, `E`, `nu` |

### 4.3 Isotropic Hardening Models

| Model Class | Description | Key Parameters |
| :--- | :--- | :--- |
| `VoceIsotropicHardeningModel` | Voce isotropic hardening | `R_inf`, `b` |
| `AdvancedVoceIsotropicHardeningModel` | Advanced Voce with memory | `Q0`, `QM`, `mu`, `b`, `q0` |
| `NoIsotropicHardeningModel` | No isotropic hardening | None |

---

## 5. Model Creation Examples

### 5.1 Loading from Files

```python
from Module_material_models import load_and_create_model
from pathlib import Path

# Load uniaxial model
model, E, yield_stress = load_and_create_model(
    kinematic_file="materials/chaboche_params.txt",
    isotropic_file="materials/voce_params.txt",
    model_type='uniaxial'
)

# Load multiaxial model
model_3d, E, yield_stress, nu = load_and_create_model(
    kinematic_file=Path("materials/owii_params.txt"),  # Path object works too
    model_type='multiaxial'
)
```

### 5.2 Direct Model Creation

```python
from Module_material_models import (
    ChabocheModelUniAxial,
    VoceIsotropicHardeningModel,
    NoIsotropicHardeningModel
)

# Create isotropic hardening model
isotropic = VoceIsotropicHardeningModel(R_inf=50.0, b=10.0)

# Create Chaboche model
C = [100000.0, 50000.0, 10000.0]
gamma = [1000.0, 100.0, 10.0]
model = ChabocheModelUniAxial(isotropic_model=isotropic, C=C, gamma=gamma)
```

---

## 6. Cross-Platform Compatibility

**Status**: ✅ Fully compatible with Windows, macOS, and Linux

### 6.1 Path Handling

- Uses `pathlib.Path` for robust cross-platform path operations
- Accepts both string paths and `Path` objects
- Handles Windows (`\`), Unix (`/`), and mixed path separators

### 6.2 File Encoding

- All file operations use explicit UTF-8 encoding
- Handles different line ending formats automatically:
  - Windows: `\r\n`
  - Unix/Linux: `\n`
  - Mac (legacy): `\r`

### 6.3 Testing

A comprehensive test suite (`inter_debug_cross_platform_test.py`) verifies:
- File operations with different path formats
- Path handling with nested directories
- UTF-8 encoding support
- Material model creation
- Cross-platform compatibility

**Run tests**:
```bash
python inter_debug_cross_platform_test.py
```

---

## 7. Implementation Notes

### 7.1 Implicit vs Explicit Updates

All material models support both explicit and implicit state updates:

- **Explicit**: `update_backstress(backstress, d_epsilon, sign, state_vars)`
- **Implicit**: `update_backstress_implicit(backstress, dp, sign)`

The implicit methods are used by fully implicit solvers for better stability.

### 7.2 State Variables

Material models maintain state through:
- `backstress`: Kinematic hardening variables (array or list of tensors)
- `R`: Isotropic hardening variable (scalar)
- `p`: Accumulated plastic strain (scalar)

State variables are updated through the `update_state_vars()` method.
