# 塑性求解器

用于单轴和多轴应力状态的 J2 塑性求解器，支持运动硬化模型（Chaboche、Ohno-Wang II、Abdel-Karim-Ohno、Kang 等）。

> **中文版**: 本文件 | **English**: [README.md](README.md)

## 平台支持

本包**跨平台兼容**，已在以下平台测试：
- **Linux**（Ubuntu、Debian 等）
- **Windows**（10/11）
- **macOS**（Intel 和 Apple Silicon）

所有文件路径使用 `pathlib.Path` 以确保跨平台兼容性，文件 I/O 使用 UTF-8 编码和平台无关的行尾。

## 项目结构

```
.
├── plasticity_solver/     # 主包
│   ├── material_models.py # 材料本构模型
│   └── solver.py          # 单轴和多轴求解器
├── tests/                 # 测试脚本
├── docs/                  # 文档
├── config/                # 标定 JSON 文件
└── README.md
```

## 安装

```bash
pip install -e .
```

或者在运行测试时将项目根目录添加到 `PYTHONPATH`。

## 依赖项

- numpy（跨平台）
- scipy（跨平台）
- matplotlib（跨平台）

所有依赖项都是平台无关的，可在 Linux、Windows 和 macOS 上运行。

## 文档

- **[使用指南（中文）](docs/USER_GUIDE_中文.md)**: 材料模型与求解器使用指南（中文版）
- **[User Guide (English)](docs/USER_GUIDE.md)**: Comprehensive guide on how to use material models and solvers
- **技术文档**: 查看 `docs/` 目录获取详细技术说明

## 运行测试

从项目根目录运行：

```bash
python tests/test_multiaxial_chaboche_stress_control_path.py
python tests/test_fully_implicit_chaboche_uni_vs_multi.py
python tests/test_fully_implicit_owii_uni_vs_multi.py
```

## 支持的模型

- **Chaboche**: 非线性运动硬化，线性恢复项
- **Ohno-Wang II**: 非线性运动硬化，非线性恢复项带饱和
- **Abdel-Karim-Ohno (AKO)**: 带 Macaulay 括号的多面模型
- **Kang**: 类似 AKO，但无 Macaulay 括号

积分方法：显式、IMEX、全隐式。
