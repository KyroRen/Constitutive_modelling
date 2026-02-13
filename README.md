# Plasticity Solver

J2 plasticity solver with kinematic hardening models (Chaboche, Ohno-Wang II, Abdel-Karim-Ohno, Kang, etc.) for uniaxial and multi-axial stress states.

> **中文版**: [README_中文.md](README_中文.md) | **English**: This file

## Platform Support

This package is **cross-platform compatible** and tested on:
- **Linux** (Ubuntu, Debian, etc.)
- **Windows** (10/11)
- **macOS** (Intel and Apple Silicon)

All file paths use `pathlib.Path` for cross-platform compatibility, and file I/O uses UTF-8 encoding with platform-agnostic line endings.

## Structure

```
.
├── plasticity_solver/     # Main package
│   ├── material_models.py # Material constitutive models
│   └── solver.py          # Uniaxial and multi-axial solvers
├── tests/                 # Test scripts
├── docs/                  # Documentation
├── config/                # Calibration JSON files
└── README.md
```

## Installation

```bash
pip install -e .
```

Or add the project root to `PYTHONPATH` when running tests.

## Dependencies

- numpy (cross-platform)
- scipy (cross-platform)
- matplotlib (cross-platform)

All dependencies are platform-agnostic and work on Linux, Windows, and macOS.

## Documentation

- **[User Guide (English)](docs/USER_GUIDE.md)**: Comprehensive guide on how to use material models and solvers
- **[使用指南（中文）](docs/USER_GUIDE_中文.md)**: 材料模型与求解器使用指南（中文版）
- **Technical Documentation**: See `docs/` directory for detailed technical notes

## Running Tests

From the project root:

```bash
python tests/test_multiaxial_chaboche_stress_control_path.py
python tests/test_fully_implicit_chaboche_uni_vs_multi.py
python tests/test_fully_implicit_owii_uni_vs_multi.py
```

## Supported Models

- **Chaboche**: Nonlinear kinematic hardening with linear recovery term
- **Ohno-Wang II**: Nonlinear kinematic hardening with saturation in nonlinear recovery term
- **Abdel-Karim-Ohno (AKO)**: Multi-surface with Macaulay bracket
- **Kang**: Similar to AKO, no Macaulay bracket

Integration methods: explicit, IMEX, fully-implicit.
