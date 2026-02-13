# 使用指南：材料模型与求解器

本指南介绍如何使用塑性求解器包在各种加载条件下模拟材料行为。

## 目录

1. [快速开始](#快速开始)
2. [材料模型](#材料模型)
3. [求解器](#求解器)
4. [基本使用示例](#基本使用示例)
5. [高级功能](#高级功能)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)

---

## 快速开始

### 安装

```bash
pip install -e .
```

### 最小示例

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

## 材料模型

### 概述

材料模型定义了材料的本构行为，包括：
- **运动硬化（Kinematic hardening）**：控制屈服面的平移（Chaboche、Ohno-Wang II 等）
- **各向同性硬化（Isotropic hardening）**：控制屈服面的扩展/收缩（Voce 模型）

### 可用模型

#### 单轴模型

- `ChabocheModelUniAxial`：多分量线性运动硬化
- `OhnoWangIIModelUniAxial`：带饱和的非线性运动硬化
- `AbdelKarimOhnoModelUniAxial`：带 Macaulay 括号的多面模型
- `KangModelUniAxial`：类似 AKO，但无 Macaulay 括号

#### 多轴模型

- `ChabocheModelMultiAxial`：Chaboche 的 3D 张量版本
- `OhnoWangIIModelMultiAxial`：Ohno-Wang II 的 3D 张量版本
- `AbdelKarimOhnoModelMultiAxial`：AKO 的 3D 张量版本
- `KangModelMultiAxial`：Kang 的 3D 张量版本

### 创建材料模型

#### 方法 1：直接实例化

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

#### 方法 2：从 JSON 配置文件加载

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

#### 方法 3：使用辅助函数（从文件）

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

### 模型参数

#### Chaboche 模型
- `C`：运动硬化模量数组（MPa）
- `gamma`：饱和参数数组（无量纲）
- 分量数量：`len(C) == len(gamma)`

#### Ohno-Wang II 模型
- `C`：运动硬化模量数组（MPa）
- `gamma`：饱和参数数组（无量纲）
- `m`：幂律指数数组（通常为 10-20）
- 分量数量：`len(C) == len(gamma) == len(m)`

#### 各向同性硬化（Voce）
- `R_inf`：各向同性硬化饱和值（MPa）
- `b`：速率参数（无量纲）

---

## 求解器

### 概述

求解器集成材料模型，在规定的加载条件下计算应力-应变响应。

### 可用求解器

1. **`UnifiedMaterialSolverUniAxial`**：用于 1D 单轴应力-应变分析
2. **`UnifiedMaterialSolverMultiAxial`**：用于 3D 多轴应力-应变分析

### 求解器初始化

#### 单轴求解器

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

**参数：**
- `E`：杨氏模量（MPa）
- `yield_stress`：初始屈服应力（MPa）
- `material_model`：材料模型实例（单轴）
- `method`：`"explicit"`、`"implicit"` 或 `"fully_implicit"`（默认：`"explicit"`）
- `precision`：`"standard"`、`"high"` 或 `"scientific"`（默认：`"standard"`）

#### 多轴求解器

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

**参数：**
- `E`：杨氏模量（MPa）
- `nu`：泊松比（无量纲）
- `yield_stress`：初始屈服应力（MPa）
- `material_model`：材料模型实例（多轴）
- `method`：`"explicit"`、`"implicit_explicit"` 或 `"fully_implicit"`（默认：`"fully_implicit"`）
- `precision`：`"standard"`、`"high"` 或 `"scientific"`（默认：`"standard"`）

### 积分方法

| 方法 | 描述 | 使用场景 |
|------|------|----------|
| `explicit` | 显式积分（快速，精度较低） | 快速模拟，小增量 |
| `implicit` / `implicit_explicit` | 半隐式（IMEX） | 精度与速度平衡 |
| `fully_implicit` | 全隐式（最精确，较慢） | 需要高精度，大增量 |

### 精度级别

| 精度 | 容差 | 使用场景 |
|------|------|----------|
| `standard` | ~1e-6 | 一般用途 |
| `high` | ~1e-9 | 需要高精度 |
| `scientific` | ~1e-12 | 研究、验证 |

---

## 基本使用示例

### 示例 1：单轴应变控制模拟

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

### 示例 2：单轴循环加载

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

### 示例 3：单轴应力控制模拟

```python
# Generate stress history
stress_history = np.linspace(0, 400, 200)

# Run stress-controlled simulation
strains, stresses = solver.run_stress_controlled(stress_history)

# Note: stresses may differ from stress_history due to convergence
print(f"Target stress: {stress_history[-1]:.2f} MPa")
print(f"Actual stress: {stresses[-1]:.2f} MPa")
```

### 示例 4：多轴应变控制模拟

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

### 示例 5：多轴应力控制模拟

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

### 示例 6：从 JSON 配置加载

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

## 高级功能

### 状态管理

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

### 单步计算

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

### 多轴单步计算

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

### 方法信息

```python
# Get information about current solver settings
info = solver.get_method_info()
print(f"Method: {info['method']}")
print(f"Precision: {info['precision']}")
print(f"Available methods: {info['available_methods']}")
```

### 自定义应变/应力历史

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

## 最佳实践

### 1. 方法选择

- **使用 `fully_implicit`** 适用于：
  - 高精度要求
  - 大应变增量
  - 多周期循环加载
  - 研究和验证

- **使用 `explicit`** 适用于：
  - 快速初步模拟
  - 小应变增量（< 0.001）
  - 速度比精度更重要时

- **使用 `implicit_explicit` (IMEX)** 适用于：
  - 精度与速度平衡
  - 中等应变增量

### 2. 精度选择

- **使用 `standard`** 适用于：
  - 一般工程应用
  - 快速模拟

- **使用 `high`** 适用于：
  - 详细分析
  - 与实验数据对比
  - 生产使用中最常见的选择

- **使用 `scientific`** 适用于：
  - 研究应用
  - 验证研究
  - 数值精度至关重要时

### 3. 应变增量大小

- **单轴**：显式方法保持增量 < 0.01，隐式方法 < 0.02
- **多轴**：保持增量 < 0.005 以确保稳定性
- 对于大增量，求解器会自动处理子步

### 4. 状态重置

在独立模拟之间始终重置求解器状态：

```python
solver.reset_state()  # Reset before new simulation
stresses, strains = solver.run_strain_controlled(new_strain_history)
```

### 5. 错误处理

```python
try:
    stresses, strains = solver.run_stress_controlled(stress_history)
except RuntimeError as e:
    print(f"Simulation failed: {e}")
    print(f"Last successful point: {len(stresses)}")
```

### 6. 内存管理

对于非常长的模拟，考虑分块处理：

```python
chunk_size = 1000
for i in range(0, len(full_strain_history), chunk_size):
    chunk = full_strain_history[i:i+chunk_size]
    stresses_chunk, strains_chunk = solver.run_strain_controlled(chunk)
    # Process chunk results
    # State is automatically maintained between chunks
```

---

## 故障排除

### 常见问题

#### 1. 收敛失败

**问题**：求解器无法收敛，特别是在应力控制模式下。

**解决方案**：
- 减小应变/应力增量
- 使用 `fully_implicit` 方法
- 提高精度级别
- 检查材料参数（非常大的 C 或 gamma 值可能导致问题）

```python
# Try with smaller increments
strain_history = np.linspace(0, 0.01, 500)  # More points = smaller increments
```

#### 2. 结果不合理

**问题**：应力或应变似乎不正确。

**解决方案**：
- 验证材料参数（单位：应力/模量为 MPa）
- 检查 E 和 yield_stress 是否一致
- 确保应变增量合理
- 在新模拟前重置求解器状态

#### 3. 性能缓慢

**问题**：模拟耗时过长。

**解决方案**：
- 初步运行使用 `explicit` 方法
- 降低精度级别
- 减少历史记录中的点数
- 对于多轴，考虑单轴近似是否足够

#### 4. 状态未重置

**问题**：之前的模拟影响当前结果。

**解决方案**：
```python
solver.reset_state()  # Always reset before new simulation
```

#### 5. 多轴应力控制问题

**问题**：应力控制的多轴模拟失败或给出意外结果。

**解决方案**：
- 确保应力张量是对称的
- 如果可能，使用应变控制模式（更稳定）
- 检查目标应力是否物理可实现
- 使用更小的应力增量

### 获取帮助

- 查看 `tests/` 目录中的测试脚本以获取工作示例
- 查看 `docs/` 目录中的文档
- 验证材料参数是否匹配预期格式
- 确保所有依赖项已正确安装

---

## 总结

本指南涵盖了使用塑性求解器包的基本方面：

1. **材料模型**：定义本构行为（Chaboche、OWII 等）
2. **求解器**：集成模型以计算响应（单轴/多轴）
3. **方法**：根据精度/速度需求选择积分方法
4. **精度**：选择数值精度容差级别
5. **模拟**：运行应变控制或应力控制分析

更多详细信息，请参考：
- `tests/` 目录中的测试脚本
- `docs/` 目录中的技术文档
- `plasticity_solver/` 模块中的源代码文档
