[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# SynReal - Newton

This repository is a fork of [newton-physics/newton](https://github.com/newton-physics/newton), maintained by [Style3D (Zhejiang Linctex Digital Technology)](https://www.style3d.com/). The `style3d` branch extends the upstream Newton codebase with:

- **`SolverStyle3DPro`** — A high-fidelity cloth solver built on the SynReal simulation SDK (`synreal-sim`), supporting anisotropic material models, garment–avatar collision, and GPU-accelerated solving.
- **Polyscope Viewer** — An interactive 3D viewer based on [Polyscope](https://polyscope.run/), supporting real-time cloth rendering and mouse-drag interaction.
- **SynReal simulation examples** — Covering garment-on-avatar simulation, robot hand–cloth interaction, and direct `synreal-sim` SDK usage.

The upstream Newton cloth solver (`SolverStyle3D`) and all existing Style3D-related examples (e.g., `cloth_style3d`, `cloth_h1`) from the main branch are retained in this fork.

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_style3d.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_style3d.jpg" alt="Cloth Style3D">
      </a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_h1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_h1.jpg" alt="Cloth H1">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>python -m newton.examples cloth_style3d</code><br>
      Anisotropic garment on a static avatar using <code>SolverStyle3D</code>.
    </td>
    <td align="center">
      <code>python -m newton.examples cloth_h1</code><br>
      H1 robot in a jacket, driven by IK — cloth powered by <code>SolverStyle3D</code>.
    </td>
  </tr>
</table>

## Requirements

- **Python** 3.10+
- **OS:** Linux (x86-64, aarch64) or Windows (x86-64)
- **GPU:** NVIDIA GPU (Maxwell or newer), driver 545 or newer (CUDA 12)
- **synreal-sim:** SynReal simulation SDK (requires a license — contact us at [SynReal](https://github.com/SynReal) to obtain access)
- **polyscope:** `pip install polyscope`

## Quickstart

```bash
# Clone this fork and switch to the style3d branch
git clone https://github.com/SynReal/newton.git
cd newton
git checkout style3d

# Install dependencies
pip install "newton[examples]"
pip install polyscope

# Run the Style3D Pro cloth simulation example
python style3d/examples/example_style3d_pro.py
```

## Style3D Pro Solver

### Overview

`SolverStyle3DPro` is a Newton solver extension that wraps the SynReal commercial simulation engine (`synreal-sim`), providing industrial-grade cloth simulation:

- Anisotropic elastic model (`tri_aniso_ke`, `edge_aniso_ke`)
- Cloth–rigid-body collision detection (garment on avatar)
- GPU-accelerated solving (`enable_gpu = True` in `WorldAttrib`)
- Mouse-drag interaction (`enable_mouse_dragging=True`)
- Requires Style3D SDK license login via `sim.login()`

### Basic Usage

```python
import newton
import warp as wp
from style3d.style3d_pro import SolverStyle3DPro

# Build model
builder = newton.Style3DModelBuilder(up_axis=newton.Axis.Z)
# ... add cloth mesh, rigid bodies, etc. ...
model = builder.finalize()

# Create solver
solver = SolverStyle3DPro(
    model=model,
    iterations=4,
    enable_mouse_dragging=True,
)
solver.precompute(builder)

# Simulation loop
state_0 = model.state()
state_1 = model.state()
for _ in range(num_steps):
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```

### License Login

`SolverStyle3DPro` requires Style3D SDK authorization before running:

```python
import synreal_sim as sim

# Option 1: interactive prompt
sim.login(username, password, True, None)

# Option 2: read from config file (recommended)
import json
with open("simulation_login.json") as f:
    cred = json.load(f)
sim.login(cred["name"], cred["pass_word"], True, None)
```

See [`style3d/simulation_login_template.json`](style3d/simulation_login_template.json) for the config file format.

## Polyscope Viewer

This branch provides two [Polyscope](https://polyscope.run/)-based visualization classes:

| Class | File | Description |
|-------|------|-------------|
| `Viewer` | `style3d/viewer/viewer.py` | General-purpose Polyscope renderer for cloth meshes, rigid bodies, and particles |
| `ViewerNewton` | `style3d/viewer/viewer_newton.py` | Extends both `Viewer` and Newton's `ViewerBase`; integrates with the Newton simulation loop |

### Basic Usage

```python
from style3d.viewer.viewer_newton import ViewerNewton
import newton

viewer = ViewerNewton(up_axis=newton.Axis.Z)
viewer.render(example)  # calls step() each frame and refreshes rendering
```

### Features

- **Real-time cloth rendering:** Triangle mesh with per-frame vertex updates
- **Rigid body rendering:** Multiple bodies with live transform updates
- **Particle rendering:** Point-cloud display
- **Mouse drag:** Cloth picking and dragging via the Polyscope pick interface
- **Pause / resume:** ImGui control panel
- **FPS counter:** Live frame-rate display

## Examples

### Style3D Cloth Examples (upstream Newton)

These examples ship in the upstream Newton codebase and are retained in this fork. They use the `SolverStyle3D` solver that is part of Newton core.

- `python -m newton.examples cloth_style3d` — Anisotropic garment on a static avatar using `SolverStyle3D`.
- `python -m newton.examples cloth_h1` — H1 robot in a jacket, driven by IK — cloth powered by `SolverStyle3D`.

### Style3D Branch Examples

These examples are added in this branch and live under `style3d/examples/`.

#### Style3D (basic garment simulation)

[`style3d/examples/example_style3d.py`](style3d/examples/example_style3d.py)

Loads a garment and avatar from USD assets and runs anisotropic cloth simulation using the Newton built-in `SolverStyle3D`, rendered with `ViewerNewton`.

```bash
python style3d/examples/example_style3d.py
```

#### Style3D Pro (high-fidelity cloth simulation)

[`style3d/examples/example_style3d_pro.py`](style3d/examples/example_style3d_pro.py)

Uses `SolverStyle3DPro` to drive the Style3D commercial solver engine, loading a women's sweatshirt USD asset. Supports GPU acceleration and mouse-drag interaction. Requires Style3D SDK license login before running.

```bash
python style3d/examples/example_style3d_pro.py
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `tri_aniso_ke` | Triangle anisotropic stiffness (warp / weft / shear) |
| `edge_aniso_ke` | Bending anisotropic stiffness |
| `density` | Cloth area density (kg/m²) |
| `soft_contact_ke` | Cloth–rigid-body contact stiffness |
| `enable_gpu` | Enable GPU-accelerated solving |

#### Hand Push Cloth (robot–cloth interaction)

[`style3d/examples/example_hand_push_cloth.py`](style3d/examples/example_hand_push_cloth.py)

Couples a robot hand joint model (URDF/USDA) with Style3D cloth simulation to demonstrate a robot end-effector pushing a cloth object.

```bash
python style3d/examples/example_hand_push_cloth.py
```

#### 3D-Sim Co-simulation

[`style3d/examples/example_sim3dsim.py`](style3d/examples/example_sim3dsim.py)

Directly invokes the `synreal-sim` SDK to simulate cloth and rigid bodies together, rendered via the Polyscope `Viewer`.

```bash
python style3d/examples/example_sim3dsim.py
```

## Repository Layout (style3d additions)

```
style3d/
├── __init__.py
├── simulation_login_template.json   # Login config template
├── style3d_mini/                    # Lightweight Style3D solver wrapper
│   └── style3d_mini.py
├── style3d_pro/                     # SolverStyle3DPro
│   ├── __init__.py
│   └── solver_style3d_pro.py
├── viewer/                          # Polyscope-based viewer
│   ├── __init__.py
│   ├── viewer.py                    # General-purpose Viewer base class
│   └── viewer_newton.py             # Newton-integrated Viewer
└── examples/                        # Branch-specific example scripts
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
