[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Newton — Style3D 分支

Newton 是基于 [NVIDIA Warp](https://github.com/NVIDIA/warp) 构建的 GPU 加速物理仿真引擎，专注于机器人研究与仿真应用。

本分支（`style3d`）由 [浙江凌迪数字科技（Style3D）](https://www.style3d.com/) 维护，在 Newton 主线的基础上集成了以下扩展：

- **`SolverStyle3DPro`** — 基于 Style3D 仿真 SDK（`style3dsim`）的高保真布料求解器，支持各向异性材质、服装-人体交互及 GPU 加速；
- **Polyscope Viewer** — 基于 [Polyscope](https://polyscope.run/) 的交互式 3D 可视化工具，支持实时布料渲染与鼠标拖拽交互；
- **Style3D 仿真示例** — 覆盖服装穿着仿真、手推布料、3D-Sim 联合仿真等场景。

## 依赖要求

- **Python** 3.10+
- **OS：** Linux (x86-64、aarch64) 或 Windows (x86-64)
- **GPU：** NVIDIA GPU（Maxwell 及以上），驱动版本 ≥ 545（CUDA 12）
- **style3dsim：** Style3D 仿真 SDK（需向 Style3D 申请授权，须登录后使用）
- **polyscope：** `pip install polyscope`

## 快速开始

```bash
# 克隆仓库并切换到 style3d 分支
git clone https://github.com/SynReal/newton.git
cd newton
git checkout style3d

# 安装依赖
pip install "newton[examples]"
pip install polyscope

# 运行 Style3D Pro 布料仿真示例
python style3d/examples/example_style3d_pro.py
```

## Style3D Pro 求解器

### 简介

`SolverStyle3DPro` 是 Newton 的扩展求解器，封装了 Style3D 商业仿真引擎（`style3dsim`），提供工业级布料仿真能力：

- 支持各向异性弹性模型（`tri_aniso_ke`、`edge_aniso_ke`）
- 支持服装与刚体（人体模型）之间的碰撞检测
- 支持 GPU 加速求解（需在 `WorldAttrib` 中开启 `enable_gpu = True`）
- 支持鼠标拖拽交互（`enable_mouse_dragging=True`）
- 需要通过 `sim.login()` 进行 Style3D SDK 授权登录

### 基本用法

```python
import newton
import warp as wp
from style3d.style3d_pro import SolverStyle3DPro
from style3d.viewer.viewer_newton import ViewerNewton

# 构建模型
builder = newton.Style3DModelBuilder(up_axis=newton.Axis.Z)
# ... 添加布料网格、刚体等 ...
model = builder.finalize()

# 创建求解器
solver = SolverStyle3DPro(
    model=model,
    iterations=4,
    enable_mouse_dragging=True,
)
solver.precompute(builder)

# 仿真循环
state_0 = model.state()
state_1 = model.state()
for _ in range(num_steps):
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

### 授权登录

`SolverStyle3DPro` 依赖 Style3D SDK 授权，运行前需要登录：

```python
import style3dsim as sim

# 方式一：交互式输入
sim.login(username, password, True, None)

# 方式二：从配置文件读取（推荐）
import json
with open("simulation_login.json") as f:
    cred = json.load(f)
sim.login(cred["name"], cred["pass_word"], True, None)
```

`simulation_login.json` 格式参考 [`style3d/simulation_login_template.json`](style3d/simulation_login_template.json)。

## Polyscope Viewer

### 简介

本分支提供了两个基于 [Polyscope](https://polyscope.run/) 的可视化类：

| 类名 | 文件 | 说明 |
|------|------|------|
| `Viewer` | `style3d/viewer/viewer.py` | 通用 Polyscope 渲染器，支持布料网格、刚体、粒子的实时渲染 |
| `ViewerNewton` | `style3d/viewer/viewer_newton.py` | 继承自 `Viewer` 和 Newton `ViewerBase`，与 Newton 仿真循环无缝集成 |

### 基本用法

```python
from style3d.viewer.viewer_newton import ViewerNewton
import newton

viewer = ViewerNewton(up_axis=newton.Axis.Z)

# 绑定仿真回调，viewer 会在每帧调用 step 并刷新渲染
viewer.render(example)
```

### 功能特性

- **实时布料渲染：** 三角面网格，支持逐帧更新顶点位置
- **刚体渲染：** 支持多刚体、变换矩阵实时更新
- **粒子渲染：** 支持粒子点云显示
- **鼠标拖拽：** 通过 Polyscope 拾取接口实现布料拖拽交互
- **暂停/继续：** ImGui UI 面板控制仿真播放
- **FPS 统计：** 实时帧率显示

## 示例

示例代码位于 [`style3d/examples/`](style3d/examples/)。

### Style3D（基础布料仿真）

[`style3d/examples/example_style3d.py`](style3d/examples/example_style3d.py)

使用 Newton 内置求解器（`SolverStyle3D`）加载服装与人体 USD 资产，进行各向异性布料仿真，通过 `ViewerNewton` 实时渲染。

```bash
python style3d/examples/example_style3d.py
```

### Style3D Pro（高保真布料仿真）

[`style3d/examples/example_style3d_pro.py`](style3d/examples/example_style3d_pro.py)

使用 `SolverStyle3DPro` 驱动 Style3D 商业求解引擎，加载女性上衣 USD 资产进行仿真，支持 GPU 加速与鼠标拖拽交互。运行前需完成 Style3D SDK 授权登录。

```bash
python style3d/examples/example_style3d_pro.py
```

**主要参数说明：**

| 参数 | 说明 |
|------|------|
| `tri_aniso_ke` | 三角形各向异性刚度（warp/weft/shear） |
| `edge_aniso_ke` | 弯曲各向异性刚度 |
| `density` | 布料面密度（kg/m²） |
| `soft_contact_ke` | 布料-刚体接触刚度 |
| `enable_gpu` | 是否启用 GPU 求解 |

### 手推布料（机器人-布料交互）

[`style3d/examples/example_hand_push_cloth.py`](style3d/examples/example_hand_push_cloth.py)

将机器人手部关节模型（URDF/USDA）与 Style3D 布料仿真联合，演示机器人末端执行器推动布料的交互场景。

```bash
python style3d/examples/example_hand_push_cloth.py
```

### 3D-Sim 联合仿真

[`style3d/examples/example_sim3dsim.py`](style3d/examples/example_sim3dsim.py)

直接调用 `style3dsim` SDK 接口，演示布料与刚体的联合仿真，通过 Polyscope `Viewer` 渲染结果。

```bash
python style3d/examples/example_sim3dsim.py
```

## 项目结构

```
style3d/
├── __init__.py
├── simulation_login_template.json   # 登录配置模板
├── style3d_mini/                    # 轻量级 Style3D 求解器封装
│   └── style3d_mini.py
├── style3d_pro/                     # SolverStyle3DPro 求解器
│   ├── __init__.py
│   └── solver_style3d_pro.py
├── viewer/                          # Polyscope 可视化
│   ├── __init__.py
│   ├── viewer.py                    # 通用 Viewer 基类
│   └── viewer_newton.py             # Newton 集成 Viewer
└── examples/                        # 示例脚本
    ├── example_style3d.py
    ├── example_style3d_pro.py
    ├── example_hand_push_cloth.py
    ├── example_sim3dsim.py
    ├── push_cloth_wonik_allegro/
    └── push_cloth_zjrx/
```

## Contributing and Development

See the [contribution guidelines](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md) and the [development guide](https://newton-physics.github.io/newton/latest/guide/development.html) for instructions on how to contribute to Newton.

## Support and Community Discussion

For questions, please consult the [Newton documentation](https://newton-physics.github.io/newton/latest/guide/overview.html) first before creating [a discussion in the main repository](https://github.com/newton-physics/newton/discussions).

## Code of Conduct

By participating in this community, you agree to abide by the Linux Foundation [Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

## Project Governance, Legal, and Members

Please see the [newton-governance repository](https://github.com/newton-physics/newton-governance) for more information about project governance.
