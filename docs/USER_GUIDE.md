# User Guide: Using Material Models and Solvers

This guide explains how to use the plasticity solver package to simulate material behavior under various loading conditions.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Material Models](#material-models)
3. [Solvers](#solvers)
4. [Basic Usage Examples](#basic-usage-examples)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Minimal Example

```python
import numpy as np
from plasticity_solver import (
    ChabocheModelUniAxial,
    VoceIsotropicHardeningModel,
    UnifiedMaterialSolverUniAxial,
)

# Create material model
isotropic = VoceIsotropicHardeningModel(R_inf=100.0, b=10.0)
model = ChabocheModelUniAxial(
    isotropic_model=isotropic,
    C=np.array([100000.0, 50000.0]),
    gamma=np.array([1000.0, 500.0])
)

# Create solver
solver = UnifiedMaterialSolverUniAxial(
    E=200000.0,  # MPa
    yield_stress=300.0,  # MPa
    material_model=model,
    method="fully_implicit",
    precision="high"
)

# Run strain-controlled simulation
strain_history = np.linspace(0, 0.01, 100)
stresses, strains = solver.run_strain_controlled(strain_history)

print(f"Final stress: {stresses[-1]:.2f} MPa")
```

---

## Material Models

### Overview

Material models define the constitutive behavior of materials, including:
- **Kinematic hardening**: Controls how the yield surface translates (Chaboche, Ohno-Wang II, etc.)
- **Isotropic hardening**: Controls how the yield surface expands/contracts (Voce model)

### Available Models

#### Uniaxial Models

- `ChabocheModelUniAxial`: Linear kinematic hardening with multiple components
- `OhnoWangIIModelUniAxial`: Nonlinear kinematic hardening with saturation
- `AbdelKarimOhnoModelUniAxial`: Multi-surface model with Macaulay bracket
- `KangModelUniAxial`: Similar to AKO, without Macaulay bracket

#### Multiaxial Models

- `ChabocheModelMultiAxial`: 3D tensor version of Chaboche
- `OhnoWangIIModelMultiAxial`: 3D tensor version of Ohno-Wang II
- `AbdelKarimOhnoModelMultiAxial`: 3D tensor version of AKO
- `KangModelMultiAxial`: 3D tensor version of Kang

### Creating Material Models

#### Method 1: Direct Instantiation

```python
from plasticity_solver import (
    ChabocheModelUniAxial,
    VoceIsotropicHardeningModel,
    NoIsotropicHardeningModel,
)

# With isotropic hardening (Voce model)
isotropic = VoceIsotropicHardeningModel(R_inf=100.0, b=10.0)
model = ChabocheModelUniAxial(
    isotropic_model=isotropic,
    C=np.array([100000.0, 50000.0, 20000.0]),  # MPa
    gamma=np.array([1000.0, 500.0, 200.0])      # dimensionless
)

# Without isotropic hardening
no_isotropic = NoIsotropicHardeningModel()
model = ChabocheModelUniAxial(
    isotropic_model=no_isotropic,
    C=np.array([100000.0, 50000.0]),
    gamma=np.array([1000.0, 500.0])
)
```

#### Method 2: Loading from JSON Configuration

```python
from pathlib import Path
import json
from plasticity_solver import (
    ChabocheModelUniAxial,
    VoceIsotropicHardeningModel,
)

# Load from JSON file
json_path = Path("config/Steel_Chaboche.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

mat_props = data["material_properties"]
params = data["calibrated_parameters"]

# Extract parameters
E = mat_props["E_MPa"]
sigma_y = mat_props["sigy0_MPa"]
C = np.array(params["C_MPa"])
gamma = np.array(params["gamma"])
R_inf = params.get("Q_MPa", 0.0)
b = params.get("b", 0.0)

# Create models
isotropic = VoceIsotropicHardeningModel(R_inf=R_inf, b=b)
model_uni = ChabocheModelUniAxial(
    isotropic_model=isotropic,
    C=C,
    gamma=gamma
)
```

#### Method 3: Using Helper Function (from File)

```python
from pathlib import Path
from plasticity_solver.material_models import load_and_create_model

# Load uniaxial model from parameter file
model, E, yield_stress = load_and_create_model(
    kinematic_file="path/to/kinematic_params.txt",
    isotropic_file="path/to/isotropic_params.txt",  # optional
    model_type="uniaxial"
)

# Load multiaxial model
model, E, yield_stress, nu = load_and_create_model(
    kinematic_file="path/to/kinematic_params.txt",
    model_type="multiaxial"
)
```

### Model Parameters

#### Chaboche Model
- `C`: Array of kinematic hardening moduli (MPa)
- `gamma`: Array of saturation parameters (dimensionless)
- Number of components: `len(C) == len(gamma)`

#### Ohno-Wang II Model
- `C`: Array of kinematic hardening moduli (MPa)
- `gamma`: Array of saturation parameters (dimensionless)
- `m`: Array of power-law exponents (typically 10-20)
- Number of components: `len(C) == len(gamma) == len(m)`

#### Isotropic Hardening (Voce)
- `R_inf`: Saturation value of isotropic hardening (MPa)
- `b`: Rate parameter (dimensionless)

---

## Solvers

### Overview

Solvers integrate material models to compute stress-strain responses under prescribed loading conditions.

### Available Solvers

1. **`UnifiedMaterialSolverUniAxial`**: For 1D uniaxial stress-strain analysis
2. **`UnifiedMaterialSolverMultiAxial`**: For 3D multiaxial stress-strain analysis

### Solver Initialization

#### Uniaxial Solver

```python
from plasticity_solver import UnifiedMaterialSolverUniAxial

solver = UnifiedMaterialSolverUniAxial(
    E=200000.0,              # Young's modulus (MPa)
    yield_stress=300.0,      # Initial yield stress (MPa)
    material_model=model,    # Material model instance
    method="fully_implicit", # Integration method
    precision="high"         # Precision level
)
```

**Parameters:**
- `E`: Young's modulus (MPa)
- `yield_stress`: Initial yield stress (MPa)
- `material_model`: Material model instance (uniaxial)
- `method`: `"explicit"`, `"implicit"`, or `"fully_implicit"` (default: `"explicit"`)
- `precision`: `"standard"`, `"high"`, or `"scientific"` (default: `"standard"`)

#### Multiaxial Solver

```python
from plasticity_solver import UnifiedMaterialSolverMultiAxial

solver = UnifiedMaterialSolverMultiAxial(
    E=200000.0,              # Young's modulus (MPa)
    nu=0.3,                  # Poisson's ratio
    yield_stress=300.0,      # Initial yield stress (MPa)
    material_model=model,    # Material model instance (multiaxial)
    method="fully_implicit", # Integration method
    precision="high"         # Precision level
)
```

**Parameters:**
- `E`: Young's modulus (MPa)
- `nu`: Poisson's ratio (dimensionless)
- `yield_stress`: Initial yield stress (MPa)
- `material_model`: Material model instance (multiaxial)
- `method`: `"explicit"`, `"implicit_explicit"`, or `"fully_implicit"` (default: `"fully_implicit"`)
- `precision`: `"standard"`, `"high"`, or `"scientific"` (default: `"standard"`)

### Integration Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `explicit` | Explicit integration (fast, less accurate) | Quick simulations, small increments |
| `implicit` / `implicit_explicit` | Semi-implicit (IMEX) | Balanced accuracy and speed |
| `fully_implicit` | Fully implicit (most accurate, slower) | High accuracy required, large increments |

### Precision Levels

| Precision | Tolerance | Use Case |
|-----------|-----------|----------|
| `standard` | ~1e-6 | General use |
| `high` | ~1e-9 | High accuracy required |
| `scientific` | ~1e-12 | Research, validation |

---

## Basic Usage Examples

### Example 1: Uniaxial Strain-Controlled Simulation

```python
import numpy as np
from plasticity_solver import (
    ChabocheModelUniAxial,
    VoceIsotropicHardeningModel,
    UnifiedMaterialSolverUniAxial,
)

# Create model
isotropic = VoceIsotropicHardeningModel(R_inf=100.0, b=10.0)
model = ChabocheModelUniAxial(
    isotropic_model=isotropic,
    C=np.array([100000.0, 50000.0]),
    gamma=np.array([1000.0, 500.0])
)

# Create solver
solver = UnifiedMaterialSolverUniAxial(
    E=200000.0,
    yield_stress=300.0,
    material_model=model,
    method="fully_implicit",
    precision="high"
)

# Generate strain history (monotonic)
strain_history = np.linspace(0, 0.02, 200)

# Run simulation
stresses, strains = solver.run_strain_controlled(strain_history)

# Access results
print(f"Max stress: {np.max(stresses):.2f} MPa")
print(f"Plastic strain: {solver.plastic_strain:.6f}")
```

### Example 2: Uniaxial Cyclic Loading

```python
# Generate cyclic strain path
strain_history, cycle_ids, seq_ids = solver.generate_cyclic_path(
    amp_pos=0.01,      # Maximum strain
    amp_neg=-0.01,     # Minimum strain
    n_cycles=10,       # Number of cycles
    n_points=100       # Points per segment
)

# Run simulation
stresses, strains = solver.run_strain_controlled(strain_history)

# Plot results
import matplotlib.pyplot as plt
plt.plot(strains, stresses)
plt.xlabel("Strain")
plt.ylabel("Stress (MPa)")
plt.grid(True)
plt.show()
```

### Example 3: Uniaxial Stress-Controlled Simulation

```python
# Generate stress history
stress_history = np.linspace(0, 400, 200)

# Run stress-controlled simulation
strains, stresses = solver.run_stress_controlled(stress_history)

# Note: stresses may differ from stress_history due to convergence
print(f"Target stress: {stress_history[-1]:.2f} MPa")
print(f"Actual stress: {stresses[-1]:.2f} MPa")
```

### Example 4: Multiaxial Strain-Controlled Simulation

```python
import numpy as np
from plasticity_solver import (
    ChabocheModelMultiAxial,
    VoceIsotropicHardeningModel,
    UnifiedMaterialSolverMultiAxial,
)

# Create multiaxial model
isotropic = VoceIsotropicHardeningModel(R_inf=100.0, b=10.0)
model = ChabocheModelMultiAxial(
    isotropic_model=isotropic,
    C=np.array([100000.0, 50000.0]),
    gamma=np.array([1000.0, 500.0]),
    E=200000.0,
    nu=0.3
)

# Create solver
solver = UnifiedMaterialSolverMultiAxial(
    E=200000.0,
    nu=0.3,
    yield_stress=300.0,
    material_model=model,
    method="fully_implicit",
    precision="high"
)

# Generate strain tensor history (uniaxial loading in x-direction)
n_points = 200
strain_history = np.zeros((n_points, 3, 3))
strain_history[:, 0, 0] = np.linspace(0, 0.01, n_points)

# Run simulation
stress_tensors, strain_tensors = solver.run_strain_controlled(strain_history)

# Extract axial stress component
sigma_11 = stress_tensors[:, 0, 0]
epsilon_11 = strain_tensors[:, 0, 0]

print(f"Max sigma_11: {np.max(sigma_11):.2f} MPa")
```

### Example 5: Multiaxial Stress-Controlled Simulation

```python
# Generate stress tensor history
n_points = 200
stress_history = np.zeros((n_points, 3, 3))
stress_history[:, 0, 0] = np.linspace(0, 400, n_points)  # Axial stress
stress_history[:, 2, 0] = np.linspace(0, 200, n_points)  # Shear stress

# Run simulation
strain_tensors, stress_tensors = solver.run_stress_controlled(stress_history)

# Extract components
epsilon_11 = strain_tensors[:, 0, 0]
gamma_31 = 2 * strain_tensors[:, 2, 0]  # Engineering shear strain
```

### Example 6: Loading from JSON Configuration

```python
from pathlib import Path
import json
import numpy as np
from plasticity_solver import (
    ChabocheModelUniAxial,
    VoceIsotropicHardeningModel,
    UnifiedMaterialSolverUniAxial,
)

# Load configuration
json_path = Path("config/Steel_Chaboche.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

mat_props = data["material_properties"]
params = data["calibrated_parameters"]

# Extract parameters
E = mat_props["E_MPa"]
sigma_y = mat_props["sigy0_MPa"]
C = np.array(params["C_MPa"])
gamma = np.array(params["gamma"])
R_inf = params.get("Q_MPa", 0.0)
b = params.get("b", 0.0)

# Create model and solver
isotropic = VoceIsotropicHardeningModel(R_inf=R_inf, b=b)
model = ChabocheModelUniAxial(isotropic_model=isotropic, C=C, gamma=gamma)

solver = UnifiedMaterialSolverUniAxial(
    E=E,
    yield_stress=sigma_y,
    material_model=model,
    method="fully_implicit",
    precision="high"
)

# Run simulation
strain_history = np.linspace(0, 0.01, 100)
stresses, strains = solver.run_strain_controlled(strain_history)
```

---

## Advanced Features

### State Management

```python
# Reset solver state to initial conditions
solver.reset_state()

# Access current state variables
print(f"Current stress: {solver.current_stress:.2f} MPa")
print(f"Plastic strain: {solver.plastic_strain:.6f}")
print(f"Backstress: {solver.backstress}")
print(f"Accumulated plastic strain: {solver.p:.6f}")
print(f"Isotropic hardening: {solver.R:.2f} MPa")
```

### Single-Step Computations

```python
# Compute stress for a given strain (uniaxial)
strain = 0.005
stress = solver.compute_stress(strain)
print(f"Stress at strain {strain}: {stress:.2f} MPa")

# Compute strain for a given stress (uniaxial)
target_stress = 400.0
strain = solver.compute_strain(target_stress)
print(f"Strain at stress {target_stress} MPa: {strain:.6f}")
```

### Multiaxial Single-Step Computations

```python
# Compute stress tensor for a given strain tensor
strain_tensor = np.array([[0.005, 0, 0],
                          [0, -0.0015, 0],
                          [0, 0, -0.0015]])
stress_tensor = solver.compute_stress(strain_tensor)
print(f"Stress tensor:\n{stress_tensor}")

# Compute strain tensor for a given stress tensor
target_stress_tensor = np.array([[400.0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]])
strain_tensor = solver.compute_strain(target_stress_tensor)
print(f"Strain tensor:\n{strain_tensor}")
```

### Method Information

```python
# Get information about current solver settings
info = solver.get_method_info()
print(f"Method: {info['method']}")
print(f"Precision: {info['precision']}")
print(f"Available methods: {info['available_methods']}")
```

### Custom Strain/Stress Histories

```python
# Custom uniaxial strain history
strain_history = np.array([
    0.0, 0.002, 0.005, 0.01,  # Loading
    0.008, 0.005, 0.002, 0.0,  # Unloading
    -0.002, -0.005, -0.01,     # Reverse loading
    -0.008, -0.005, -0.002, 0.0 # Reverse unloading
])

stresses, strains = solver.run_strain_controlled(strain_history)

# Custom multiaxial stress history
n_points = 100
stress_history = np.zeros((n_points, 3, 3))
stress_history[:, 0, 0] = np.sin(np.linspace(0, 4*np.pi, n_points)) * 300
stress_history[:, 2, 0] = np.cos(np.linspace(0, 4*np.pi, n_points)) * 200

strain_tensors, stress_tensors = solver.run_stress_controlled(stress_history)
```

---

## Best Practices

### 1. Method Selection

- **Use `fully_implicit`** for:
  - High accuracy requirements
  - Large strain increments
  - Cyclic loading with many cycles
  - Research and validation

- **Use `explicit`** for:
  - Quick preliminary simulations
  - Small strain increments (< 0.001)
  - When speed is more important than accuracy

- **Use `implicit_explicit` (IMEX)** for:
  - Balanced accuracy and speed
  - Moderate strain increments

### 2. Precision Selection

- **Use `standard`** for:
  - General engineering applications
  - Quick simulations

- **Use `high`** for:
  - Detailed analysis
  - Comparison with experimental data
  - Most common choice for production use

- **Use `scientific`** for:
  - Research applications
  - Validation studies
  - When numerical precision is critical

### 3. Strain Increment Size

- **Uniaxial**: Keep increments < 0.01 for explicit, < 0.02 for implicit
- **Multiaxial**: Keep increments < 0.005 for stability
- Use substeps automatically handled by the solver for large increments

### 4. State Reset

Always reset solver state between independent simulations:

```python
solver.reset_state()  # Reset before new simulation
stresses, strains = solver.run_strain_controlled(new_strain_history)
```

### 5. Error Handling

```python
try:
    stresses, strains = solver.run_stress_controlled(stress_history)
except RuntimeError as e:
    print(f"Simulation failed: {e}")
    print(f"Last successful point: {len(stresses)}")
```

### 6. Memory Management

For very long simulations, consider processing in chunks:

```python
chunk_size = 1000
for i in range(0, len(full_strain_history), chunk_size):
    chunk = full_strain_history[i:i+chunk_size]
    stresses_chunk, strains_chunk = solver.run_strain_controlled(chunk)
    # Process chunk results
    # State is automatically maintained between chunks
```

---

## Troubleshooting

### Common Issues

#### 1. Convergence Failures

**Problem**: Solver fails to converge, especially in stress-controlled mode.

**Solutions**:
- Reduce strain/stress increments
- Use `fully_implicit` method
- Increase precision level
- Check material parameters (very large C or gamma values can cause issues)

```python
# Try with smaller increments
strain_history = np.linspace(0, 0.01, 500)  # More points = smaller increments
```

#### 2. Unrealistic Results

**Problem**: Stresses or strains seem incorrect.

**Solutions**:
- Verify material parameters (units: MPa for stress/modulus)
- Check that E and yield_stress are consistent
- Ensure strain increments are reasonable
- Reset solver state before new simulation

#### 3. Slow Performance

**Problem**: Simulations take too long.

**Solutions**:
- Use `explicit` method for preliminary runs
- Reduce precision level
- Reduce number of points in history
- For multiaxial, consider if uniaxial approximation is sufficient

#### 4. State Not Resetting

**Problem**: Previous simulation affects current results.

**Solution**:
```python
solver.reset_state()  # Always reset before new simulation
```

#### 5. Multiaxial Stress Control Issues

**Problem**: Stress-controlled multiaxial simulation fails or gives unexpected results.

**Solutions**:
- Ensure stress tensor is symmetric
- Use strain-controlled mode if possible (more stable)
- Check that target stresses are physically achievable
- Use smaller stress increments

### Getting Help

- Check test scripts in `tests/` directory for working examples
- Review documentation in `docs/` directory
- Verify material parameters match expected format
- Ensure all dependencies are installed correctly

---

## Summary

This guide covers the essential aspects of using the plasticity solver package:

1. **Material Models**: Define constitutive behavior (Chaboche, OWII, etc.)
2. **Solvers**: Integrate models to compute responses (uniaxial/multiaxial)
3. **Methods**: Choose integration method based on accuracy/speed needs
4. **Precision**: Select tolerance level for numerical accuracy
5. **Simulations**: Run strain-controlled or stress-controlled analyses

For more detailed information, refer to:
- Test scripts in `tests/` directory
- Technical documentation in `docs/` directory
- Source code documentation in `plasticity_solver/` modules
