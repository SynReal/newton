# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..solver import SolverBase
from .builder import PDMatrixBuilder
from .collision import Collision
from .kernels import (
    accumulate_dragging_pd_diag_kernel,
    eval_aero_force_kernel,
    init_inertia_warm_start_kernel,
    init_inertia_warm_start_free_kernel,
    init_inertia_warm_start_free_v_kernel,
    init_accel_warm_start_kernel,
    init_accel_f_warm_start_kernel,
    iter2c_contact_force_delta_kernel,
    iter2_gate_count_kernel,
    iter2_gate_peak_kernel,
    iter2_gate_mark_ee_kernel,
    iter2_gate_mark_soft_kernel,
    iter2_gate_mark_tri_sdf_kernel,
    iter2_gate_mark_vf_kernel,
    eval_bend_kernel,
    eval_drag_force_kernel,
    eval_stretch_kernel,
    init_rhs_kernel,
    init_step_kernel,
    nonlinear_step_kernel,
    prepare_jacobi_preconditioner_kernel,
    prepare_jacobi_preconditioner_no_contact_hessian_kernel,
    update_velocity,
)
from .linear_solver import FusedTranslationPrecond, PcgSolver, SparseMatrixELL

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency


@wp.kernel
def _accumulate_translation_preconditioner_kernel(
    dt: float,
    residual: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    contact_hessian_diags: wp.array[wp.mat33],
    # outputs
    coarse_rhs: wp.array[wp.vec3],
    coarse_diag: wp.array[wp.vec3],
):
    tid = wp.tid()
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        comp_idx = wp.min(wp.max(particle_component[tid], 0), coarse_diag.shape[0] - 1)
        mass_diag = particle_masses[tid] / (dt * dt)
        contact_hess = contact_hessian_diags[tid]
        wp.atomic_add(coarse_rhs, comp_idx, residual[tid])
        wp.atomic_add(
            coarse_diag,
            comp_idx,
            wp.vec3(
                mass_diag + contact_hess[0, 0],
                mass_diag + contact_hess[1, 1],
                mass_diag + contact_hess[2, 2],
            ),
        )


@wp.kernel
def _apply_translation_preconditioner_kernel(
    coarse_rhs: wp.array[wp.vec3],
    coarse_diag: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    # outputs
    z: wp.array[wp.vec3],
):
    tid = wp.tid()
    comp_idx = wp.min(wp.max(particle_component[tid], 0), coarse_diag.shape[0] - 1)
    denom = coarse_diag[comp_idx]
    rhs = coarse_rhs[comp_idx]
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        correction = wp.vec3(0.0)
        if denom[0] > 0.0:
            correction[0] = rhs[0] / denom[0]
        if denom[1] > 0.0:
            correction[1] = rhs[1] / denom[1]
        if denom[2] > 0.0:
            correction[2] = rhs[2] / denom[2]
        z[tid] = z[tid] + correction


# --------------------------------------------------------------------- ITER7
# Bucketed two-level reduction for the translation preconditioner
# (`tp_reduce` = "legacy" | "bucket").
#
# `_accumulate_translation_preconditioner_kernel` has EVERY active particle
# atomic-add into one slot per connected component.  On a single-component
# cloth that is one address taking all N atomics.  The bucketed form is the
# same one `coarse_correction` already uses: each particle adds into
# `tid % _TP_BUCKETS` of its component, then a second pass sums the buckets.
#
# The accumulators are identical to mode 1 of the coarse correction --
# sum r (3) + sum a (3), a_i = m_i/dt^2 + H_contact,i -- so this reuses
# `_cc_reduce_kernel` verbatim for the second level.
#
# NOT bit-identical to "legacy": float addition is not associative and the
# summation order changes.  That is why it sits behind a switch whose default
# is "legacy".
_TP_BUCKETS = 256
_TP_NACC = 6


@wp.kernel
def _tp_bucket_acc_kernel(
    dt: float,
    residual: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    contact_hessian_diags: wp.array[wp.mat33],
    n_buckets: int,
    n_comp: int,
    # outputs
    partial: wp.array[float],
):
    tid = wp.tid()
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        comp_idx = wp.min(wp.max(particle_component[tid], 0), n_comp - 1)
        base = (comp_idx * n_buckets + (tid % n_buckets)) * 6
        r = residual[tid]
        mass_diag = particle_masses[tid] / (dt * dt)
        h = contact_hessian_diags[tid]
        wp.atomic_add(partial, base + 0, r[0])
        wp.atomic_add(partial, base + 1, r[1])
        wp.atomic_add(partial, base + 2, r[2])
        wp.atomic_add(partial, base + 3, mass_diag + h[0, 0])
        wp.atomic_add(partial, base + 4, mass_diag + h[1, 1])
        wp.atomic_add(partial, base + 5, mass_diag + h[2, 2])


@wp.kernel
def _tp_bucket_finish_kernel(
    total: wp.array[float],
    # outputs
    coarse_rhs: wp.array[wp.vec3],
    coarse_diag: wp.array[wp.vec3],
):
    c = wp.tid()
    coarse_rhs[c] = wp.vec3(total[c * 6 + 0], total[c * 6 + 1], total[c * 6 + 2])
    coarse_diag[c] = wp.vec3(total[c * 6 + 3], total[c * 6 + 4], total[c * 6 + 5])


# --------------------------------------------------------------------- ITER3
# Rigid-mode Galerkin coarse correction (`coarse_correction` = 0 / 1 / 6).
#
# Runs ONCE per nonlinear iteration, on the PCG *initial guess*, OUTSIDE the PCG
# loop (so it is not a preconditioner and does not change the Krylov space).
# `A_i` is the per-particle diagonal block `m_i/dt^2 * I + H_contact,i` -- the
# same approximation `_accumulate_translation_preconditioner_kernel` makes,
# and it is exact for the translation mode because the PD elastic matrix
# annihilates a rigid translation.  With `r` the residual at `x0 = 0` (= rhs):
#
#   mode 1:  t_c = (sum_i r_i) ./ (sum_i a_i)              (componentwise)
#   mode 6:  [ D   -C  ] [t]   [ sum_i r_i        ]
#            [-C^T  S  ] [w] = [ sum_i q_i x r_i  ] ,  q_i = x_i - centroid_c
#            D = diag(sum a_i), C = sum diag(a_i) skew(q_i),
#            S = sum skew(q_i)^T diag(a_i) skew(q_i)
#   displacement u_i = t + w x q_i.
#
# The 6x6 is solved by its Schur complement onto w (D is diagonal), so only a
# 3x3 inverse is needed and everything stays inside one device thread per
# component.  Reductions are bucketed (`_CC_BUCKETS` slots per component) so no
# single address takes every particle's atomic.
_CC_BUCKETS = 256
_CC_NACC_T = 6       # mode 1: sum r (3) + sum a (3)
_CC_NACC_R = 21      # mode 6: sum r (3), sum q x r (3), sum a (3), C (6), S (6)
_CC_NACC_C = 4       # centroid: sum x (3) + count (1)


@wp.kernel
def _cc_centroid_acc_kernel(
    particle_q: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    n_buckets: int,
    n_comp: int,
    # outputs
    partial: wp.array[float],
):
    tid = wp.tid()
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        comp_idx = wp.min(wp.max(particle_component[tid], 0), n_comp - 1)
        base = (comp_idx * n_buckets + (tid % n_buckets)) * 4
        p = particle_q[tid]
        wp.atomic_add(partial, base + 0, p[0])
        wp.atomic_add(partial, base + 1, p[1])
        wp.atomic_add(partial, base + 2, p[2])
        wp.atomic_add(partial, base + 3, 1.0)


@wp.kernel
def _cc_reduce_kernel(
    partial: wp.array[float],
    n_buckets: int,
    n_acc: int,
    # outputs
    total: wp.array[float],
):
    tid = wp.tid()                       # dim = n_comp * n_acc
    comp_idx = tid // n_acc
    k = tid - comp_idx * n_acc
    s = float(0.0)
    for b in range(n_buckets):
        s = s + partial[(comp_idx * n_buckets + b) * n_acc + k]
    total[tid] = s


@wp.kernel
def _cc_centroid_finish_kernel(
    total: wp.array[float],
    # outputs
    centroid: wp.array[wp.vec3],
):
    c = wp.tid()
    n = total[c * 4 + 3]
    if n > 0.0:
        centroid[c] = wp.vec3(total[c * 4 + 0] / n, total[c * 4 + 1] / n, total[c * 4 + 2] / n)
    else:
        centroid[c] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _cc_moment_t_acc_kernel(
    dt: float,
    residual: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    contact_hessian_diags: wp.array[wp.mat33],
    n_buckets: int,
    n_comp: int,
    # outputs
    partial: wp.array[float],
):
    tid = wp.tid()
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        comp_idx = wp.min(wp.max(particle_component[tid], 0), n_comp - 1)
        base = (comp_idx * n_buckets + (tid % n_buckets)) * 6
        r = residual[tid]
        mass_diag = particle_masses[tid] / (dt * dt)
        h = contact_hessian_diags[tid]
        wp.atomic_add(partial, base + 0, r[0])
        wp.atomic_add(partial, base + 1, r[1])
        wp.atomic_add(partial, base + 2, r[2])
        wp.atomic_add(partial, base + 3, mass_diag + h[0, 0])
        wp.atomic_add(partial, base + 4, mass_diag + h[1, 1])
        wp.atomic_add(partial, base + 5, mass_diag + h[2, 2])


@wp.kernel
def _cc_moment_r_acc_kernel(
    dt: float,
    residual: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    centroid: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    contact_hessian_diags: wp.array[wp.mat33],
    n_buckets: int,
    n_comp: int,
    # outputs
    partial: wp.array[float],
):
    tid = wp.tid()
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        comp_idx = wp.min(wp.max(particle_component[tid], 0), n_comp - 1)
        base = (comp_idx * n_buckets + (tid % n_buckets)) * 21
        r = residual[tid]
        q = particle_q[tid] - centroid[comp_idx]
        mass_diag = particle_masses[tid] / (dt * dt)
        h = contact_hessian_diags[tid]
        a0 = mass_diag + h[0, 0]
        a1 = mass_diag + h[1, 1]
        a2 = mass_diag + h[2, 2]
        qr = wp.cross(q, r)
        qx = q[0]
        qy = q[1]
        qz = q[2]
        wp.atomic_add(partial, base + 0, r[0])
        wp.atomic_add(partial, base + 1, r[1])
        wp.atomic_add(partial, base + 2, r[2])
        wp.atomic_add(partial, base + 3, qr[0])
        wp.atomic_add(partial, base + 4, qr[1])
        wp.atomic_add(partial, base + 5, qr[2])
        wp.atomic_add(partial, base + 6, a0)
        wp.atomic_add(partial, base + 7, a1)
        wp.atomic_add(partial, base + 8, a2)
        # C = sum diag(a) * skew(q); six independent entries
        wp.atomic_add(partial, base + 9, a0 * qz)
        wp.atomic_add(partial, base + 10, a0 * qy)
        wp.atomic_add(partial, base + 11, a1 * qz)
        wp.atomic_add(partial, base + 12, a1 * qx)
        wp.atomic_add(partial, base + 13, a2 * qy)
        wp.atomic_add(partial, base + 14, a2 * qx)
        # S = sum skew(q)^T diag(a) skew(q); symmetric
        wp.atomic_add(partial, base + 15, a1 * qz * qz + a2 * qy * qy)
        wp.atomic_add(partial, base + 16, a0 * qz * qz + a2 * qx * qx)
        wp.atomic_add(partial, base + 17, a0 * qy * qy + a1 * qx * qx)
        wp.atomic_add(partial, base + 18, -a2 * qx * qy)
        wp.atomic_add(partial, base + 19, -a1 * qx * qz)
        wp.atomic_add(partial, base + 20, -a0 * qy * qz)


@wp.kernel
def _cc_solve_t_kernel(
    total: wp.array[float],
    # outputs
    t_out: wp.array[wp.vec3],
):
    c = wp.tid()
    o = c * 6
    t = wp.vec3(0.0, 0.0, 0.0)
    if total[o + 3] > 0.0:
        t[0] = total[o + 0] / total[o + 3]
    if total[o + 4] > 0.0:
        t[1] = total[o + 1] / total[o + 4]
    if total[o + 5] > 0.0:
        t[2] = total[o + 2] / total[o + 5]
    t_out[c] = t


@wp.kernel
def _cc_solve_r_kernel(
    total: wp.array[float],
    # outputs
    t_out: wp.array[wp.vec3],
    w_out: wp.array[wp.vec3],
):
    c = wp.tid()
    o = c * 21
    bt = wp.vec3(total[o + 0], total[o + 1], total[o + 2])
    bw = wp.vec3(total[o + 3], total[o + 4], total[o + 5])
    a0 = total[o + 6]
    a1 = total[o + 7]
    a2 = total[o + 8]
    t = wp.vec3(0.0, 0.0, 0.0)
    w = wp.vec3(0.0, 0.0, 0.0)
    if a0 > 0.0 and a1 > 0.0 and a2 > 0.0:
        c0 = total[o + 9]
        c1 = total[o + 10]
        c2 = total[o + 11]
        c3 = total[o + 12]
        c4 = total[o + 13]
        c5 = total[o + 14]
        C = wp.mat33(0.0, -c0, c1,
                     c2, 0.0, -c3,
                     -c4, c5, 0.0)
        S = wp.mat33(total[o + 15], total[o + 18], total[o + 19],
                     total[o + 18], total[o + 16], total[o + 20],
                     total[o + 19], total[o + 20], total[o + 17])
        Dinv = wp.mat33(1.0 / a0, 0.0, 0.0,
                        0.0, 1.0 / a1, 0.0,
                        0.0, 0.0, 1.0 / a2)
        CtDi = wp.transpose(C) * Dinv
        K = S - CtDi * C
        tr = K[0, 0] + K[1, 1] + K[2, 2]
        det = wp.determinant(K)
        # Scale-free rank check: a rotation mode with no lever arm (all q_i
        # collinear, or a component of one particle) leaves K singular; fall
        # back to translation only rather than inventing an omega.
        if tr > 0.0 and wp.abs(det) > 1.0e-12 * tr * tr * tr:
            w = wp.inverse(K) * (bw + CtDi * bt)
        t = Dinv * (bt + C * w)
    t_out[c] = t
    w_out[c] = w


@wp.kernel
def _cc_apply_t_kernel(
    t_in: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    n_comp: int,
    # outputs
    x0: wp.array[wp.vec3],
):
    tid = wp.tid()
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        comp_idx = wp.min(wp.max(particle_component[tid], 0), n_comp - 1)
        x0[tid] = x0[tid] + t_in[comp_idx]


@wp.kernel
def _cc_apply_r_kernel(
    t_in: wp.array[wp.vec3],
    w_in: wp.array[wp.vec3],
    centroid: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    n_comp: int,
    # outputs
    x0: wp.array[wp.vec3],
):
    tid = wp.tid()
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        comp_idx = wp.min(wp.max(particle_component[tid], 0), n_comp - 1)
        q = particle_q[tid] - centroid[comp_idx]
        x0[tid] = x0[tid] + t_in[comp_idx] + wp.cross(w_in[comp_idx], q)


########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


class SolverStyle3D(SolverBase):
    r"""Projective dynamics based cloth solver.

    References:
        1. Baraff, D. & Witkin, A. "Large Steps in Cloth Simulation."
        2. Liu, T. et al. "Fast Simulation of Mass-Spring Systems."

    Implicit-Euler method solves the following non-linear equation:

    .. math::

        (M / dt^2 + H(x)) \cdot dx &= (M / dt^2) \cdot (x_{prev} + v_{prev} \cdot dt - x) + f_{ext}(x) + f_{int}(x) \\
                                   &= (M / dt^2) \cdot (x_{prev} + v_{prev} \cdot dt + (dt^2 / M) \cdot f_{ext}(x) - x) + f_{int}(x) \\
                                   &= (M / dt^2) \cdot (x_{inertia} - x) + f_{int}(x)

    Notations:
        - :math:`M`: mass matrix
        - :math:`x`: unsolved particle position
        - :math:`H`: Hessian matrix (function of x)
        - :math:`P`: PD-approximated Hessian matrix (constant)
        - :math:`A`: :math:`M / dt^2 + H(x)` or :math:`M / dt^2 + P`
        - :math:`rhs`: Right hand side of the equation: :math:`(M / dt^2) \cdot (x_{inertia} - x) + f_{int}(x)`
        - :math:`res`: Residual: :math:`rhs - A \cdot dx_{init}`, or rhs if :math:`dx_{init} = 0`

    See Also:
        :doc:`newton.solvers.style3d </api/newton_solvers_style3d>` exposes
        helper functions that populate Style3D cloth data on a
        :class:`~newton.ModelBuilder`.

    Example:
        Build a mesh-based cloth with
        :func:`newton.solvers.style3d.add_cloth_mesh`::

            from newton.solvers import style3d

            builder = newton.ModelBuilder()
            SolverStyle3D.register_custom_attributes(builder)
            style3d.add_cloth_mesh(
                builder,
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=mesh.vertices.tolist(),
                indices=mesh.indices.tolist(),
                density=0.3,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
            )

        Or build a grid with :func:`newton.solvers.style3d.add_cloth_grid`::

            style3d.add_cloth_grid(
                builder,
                pos=wp.vec3(-0.5, 0.0, 2.0),
                rot=wp.quat_identity(),
                dim_x=64,
                dim_y=32,
                cell_x=0.1,
                cell_y=0.1,
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=0.1,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
            )

    """

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        linear_iterations: int = 10,
        drag_spring_stiff: float = 1e2,
        enable_mouse_dragging: bool = False,
        enable_translation_preconditioner: bool = False,
        vel_damping: float = 0.998,
        linear_schedule: str | None = None,
        tp_reduce: str = "legacy",
        pcg_eta: float = 0.1,
        pcg_fused: bool = False,
        inertia_warm_start: bool | str = False,
        coarse_correction: int = 0,
        inertia_warm_start_v_gate: float = 0.05,
    ):
        """
        Args:
            model: The :class:`~newton.Model` containing Style3D attributes to integrate.
            iterations: Number of non-linear iterations per step.
            linear_iterations: Number of linear iterations (currently PCG iter) per non-linear iteration.
            drag_spring_stiff: The stiffness of spring connecting barycentric-weighted drag-point and target-point.
            enable_mouse_dragging: Enable/disable dragging kernel.
            enable_translation_preconditioner: Enable a coarse per-component translation preconditioner for PCG.
            vel_damping: AERO1 -- per-step multiplicative velocity damping applied in
                ``update_velocity``.  0.998 reproduces the historical hard-coded value
                bit for bit; it is the solver's only unconditional dissipation term.
            linear_schedule: ITER1/ITER2e -- how many PCG iterations each nonlinear
                iteration gets when the translation preconditioner is OFF.

                  ``"ramp"``      default; the historical hard-coded ``min(iter + 1, 10)``,
                                  under which ``linear_iterations`` is never read on that path.
                  ``"fixed"``     honour ``linear_iterations`` on every nonlinear iteration.
                  ``"ramp_it0"``  ITER2e: ``linear_iterations`` on the FIRST nonlinear
                                  iteration only, ``min(iter + 1, 10)`` afterwards.  The
                                  warm start's residual is entirely in iteration 0, and that
                                  is the one the ramp starves (1 PCG step).  Cost at
                                  ``iterations=10``: 64 PCG steps vs the ramp's 55 (+16 %),
                                  where ``"fixed"`` costs 100 (1.8x).

                ``None`` (the default) means "stock": the ramp on the non-TP path and
                ``linear_iterations`` on the TP path, i.e. exactly what this solver did
                before ITER3.  Naming a schedule EXPLICITLY makes it apply to BOTH
                paths (ITER3 change 1) -- that is the only way the TP path can be put
                on the ramp.
            inertia_warm_start: ITER1/ITER2 -- seed the first nonlinear iteration's PCG
                guess with ``x_inertia - x_curr`` (the full free-flight step) instead of
                the stock ``v_prev * dt``.  Accepts

                  ``False`` / ``"off"``   default, bit-identical to before;
                  ``True``  / ``"all"``   ITER1: every particle, bit-identical to ITER1;
                  ``"free"``              ITER2/ITER5: only particles that carry NO contact
                                          pair this substep.  Channels: cloth-cloth broad
                                          phase (vf + ee), the tri-SDF blade AABB, and
                                          (ITER5) the particle-rigid soft contacts, i.e.
                                          the TABLE.  A contacting particle keeps the
                                          stock guess, so the 39 um free-flight offset no
                                          longer rides into the contact stack's
                                          "iteration 0 == x_prev" assumption.
                  ``"accel"``             ITER2b: constant-acceleration predictor,
                                          ``dx = v_prev*dt + a_prev*dt^2`` with
                                          ``a_prev = (v_prev - v_prev2)/dt`` measured from
                                          the two previous substeps.  Exact in free flight
                                          (``a_prev = g``) AND at rest on a support
                                          (``a_prev = 0``); no particle classification, no
                                          contact lookup, no parameter.  This is the
                                          candidate; ``"free"`` is kept only so its ITER2
                                          measurements stay reproducible.
                  ``"accel_f"``           ITER2c: ``dx = v_prev*dt + (g + f_c_prev/m)*dt^2``
                                          with ``f_c_prev`` the CONTACT force from the last
                                          nonlinear iteration of the previous substep
                                          (no elastic/bending term).  Exact in free flight
                                          (``f_c = 0`` -> ITER1's guess) and at rest on a
                                          support (``f_c = +m g`` -> the stock guess), with
                                          no velocity recursion, hence none of ``accel``'s
                                          ``(z-1)^2`` double root.
            coarse_correction: ITER3 -- rigid-mode Galerkin coarse correction applied
                to the PCG initial guess at the START of every nonlinear iteration
                (after the rhs is complete, before PCG; NOT inside the Krylov loop).

                  ``0``  default, OFF, bit-identical to before;
                  ``1``  translation only -- the same coarse operator the translation
                         preconditioner uses, but spent once per nonlinear iteration
                         on the guess instead of once per PCG step inside it;
                  ``6``  translation + rotation -- a 6x6 per connected component.

                Independent of ``inertia_warm_start``: with both on the guess is
                ``warm start + coarse correction``, neither knows about the other.
        """

        super().__init__(model)
        if not hasattr(model, "style3d"):
            raise AttributeError(
                "Style3D custom attributes are missing from the model. "
                "Call SolverStyle3D.register_custom_attributes() before building the model."
            )
        self.style3d = model.style3d
        self.collision: Collision | None = Collision(model)  # set None to disable
        # T14: the contact stack's iteration-stride knob needs to know which
        # iteration is the LAST one of a substep -- that is the iterate whose
        # solve produces the positions the substep ends on, and handing it a
        # held contact force is what breaks the grasp (measured).
        if self.collision is not None:
            self.collision.nonlinear_iterations = int(iterations)
        self.linear_iterations = linear_iterations
        self.nonlinear_iterations = iterations
        self.drag_spring_stiff = drag_spring_stiff
        self.enable_mouse_dragging = enable_mouse_dragging
        self.enable_translation_preconditioner = enable_translation_preconditioner
        # FUSE1: `pcg_fused` collapses the PCG step's element-wise kernels
        # (and, when the translation preconditioner is on, its two kernels) to
        # 3-4 launches per step.  Default OFF -> the stock path is untouched.
        self.pcg_fused = bool(pcg_fused)
        self.linear_solver = PcgSolver(model.particle_count, self.device, fused=self.pcg_fused)

        # Fixed PD matrix
        self.pd_non_diags = SparseMatrixELL()
        self.pd_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self._precompute(model)

        # Non-linear equation variables
        self.dx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.rhs = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_inertia = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

        # Static part of A_diag, full A_diag, and inverse of A_diag
        self.static_A_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self.inv_A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self.A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)

        # Coarse translation preconditioner buffers.
        self._translation_preconditioner_dt = 0.0
        self._translation_component_count, particle_component = self._build_particle_components(model)
        self._translation_particle_component = wp.array(particle_component, dtype=wp.int32, device=self.device)
        self._translation_coarse_rhs = wp.zeros(self._translation_component_count, dtype=wp.vec3, device=self.device)
        self._translation_coarse_diag = wp.zeros(self._translation_component_count, dtype=wp.vec3, device=self.device)
        self._translation_zero_contact_hessian = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self._translation_contact_hessian_diags = self._translation_zero_contact_hessian

        self.vel_damping = float(vel_damping)

        # ---------------------------------------------------------------- ITER3
        self.coarse_correction = int(coarse_correction)
        if self.coarse_correction not in (0, 1, 6):
            raise ValueError(
                f"coarse_correction must be 0, 1 or 6, got {coarse_correction!r}"
            )
        self._cc_partial = None
        self._cc_total = None
        self._cc_t = None
        self._cc_w = None
        self._cc_x0 = None
        self._cc_centroid = None
        self._cc_cpartial = None
        self._cc_ctotal = None
        if self.coarse_correction:
            _nc = self._translation_component_count
            _na = _CC_NACC_T if self.coarse_correction == 1 else _CC_NACC_R
            self._cc_partial = wp.zeros(_nc * _CC_BUCKETS * _na, dtype=float, device=self.device)
            self._cc_total = wp.zeros(_nc * _na, dtype=float, device=self.device)
            self._cc_t = wp.zeros(_nc, dtype=wp.vec3, device=self.device)
            self._cc_w = wp.zeros(_nc, dtype=wp.vec3, device=self.device)
            self._cc_x0 = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
            if self.coarse_correction == 6:
                self._cc_centroid = wp.zeros(_nc, dtype=wp.vec3, device=self.device)
                self._cc_cpartial = wp.zeros(_nc * _CC_BUCKETS * _CC_NACC_C, dtype=float,
                                             device=self.device)
                self._cc_ctotal = wp.zeros(_nc * _CC_NACC_C, dtype=float, device=self.device)

        # ITER3: `None` = "stock" -- the historical per-path behaviour (ramp off the
        # TP path, `linear_iterations` on it).  An EXPLICIT name applies to both.
        if linear_schedule is None:
            linear_schedule = "stock"
        if linear_schedule not in ("stock", "ramp", "fixed", "ramp_it0", "adaptive"):
            raise ValueError(
                "linear_schedule must be None/'stock'/'ramp'/'fixed'/'ramp_it0'/'adaptive', "
                f"got {linear_schedule!r}"
            )
        self.linear_schedule = str(linear_schedule)
        # ------------------------------------------------------------ ITER7
        if str(tp_reduce) not in ("legacy", "bucket"):
            raise ValueError(f"tp_reduce must be 'legacy' or 'bucket', got {tp_reduce!r}")
        self.tp_reduce = str(tp_reduce)
        # FUSE1: rebuilt (dt / contact Hessian) once per `solve()`.
        self._fused_tp = FusedTranslationPrecond()
        self._fused_tp.enabled = 0
        self._fused_tp.n_comp = int(self._translation_component_count)
        self._fused_tp.dt = 0.0
        self._fused_tp.particle_component = self._translation_particle_component
        self._fused_tp.particle_masses = model.particle_mass
        self._fused_tp.particle_flags = model.particle_flags
        self._fused_tp.contact_hessian_diags = self._translation_zero_contact_hessian
        self._fused_tp.coarse_rhs = self._translation_coarse_rhs
        self._fused_tp.coarse_diag = self._translation_coarse_diag
        self.pcg_eta = float(pcg_eta)
        # `adaptive` is the ONLY schedule that arms the PCG stop test; every other
        # schedule passes eta = 0 into PcgSolver.solve and keeps the historical
        # fixed-trip-count loop, bit for bit.
        self._pcg_eta_active = self.pcg_eta if self.linear_schedule == "adaptive" else 0.0
        # Bucket scratch: allocated only when the bucket path is selected.
        self._tp_partial = None
        self._tp_total = None

        # ITER2: `inertia_warm_start` grew a third value.  The two historical
        # ones keep their exact meaning (and `self.inertia_warm_start` stays a
        # truthy bool for anything that looked at it).
        _iws = inertia_warm_start
        if isinstance(_iws, str):
            _mode = _iws.strip().lower()
            if _mode in ("", "0", "off", "false", "no"):
                _mode = "off"
            elif _mode in ("1", "on", "true", "yes", "all"):
                _mode = "all"
        else:
            _mode = "all" if bool(_iws) else "off"
        if _mode not in ("off", "all", "free", "free_v", "accel", "accel_f"):
            raise ValueError(
                "inertia_warm_start must be False/True/'off'/'all'/'free'/'free_v'/"
                "'accel'/'accel_f', "
                f"got {inertia_warm_start!r}"
            )
        self.inertia_warm_start_mode = _mode
        self.inertia_warm_start = _mode != "off"
        # ITER2 gate: 1 = this particle carries a contact pair this substep.
        # Allocated only in 'free' mode, so the other two paths do not even pay
        # the allocation.
        self.iter2_gate = None
        self.iter2_gate_stat = None
        self.iter2_gate_peak = None
        # ITER2b: the substep-before-last velocity, so the predictor can measure
        # a_prev.  Primed on the first step with the current velocity, which
        # makes a_prev = 0 there, i.e. the first substep degenerates to the
        # stock `dx = v_prev*dt`.
        self.v_prev2 = None
        self._accel_primed = False
        if _mode == "accel":
            self.v_prev2 = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        # ITER2c: the previous substep's contact force.  Zero on the first
        # substep, which makes the guess ITER1's.
        self.f_contact = None
        if _mode == "accel_f":
            self.f_contact = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        # ITER5c: the velocity gate.  Read only in mode "free_v".
        self.inertia_warm_start_v_gate = float(inertia_warm_start_v_gate)
        if _mode in ("free", "free_v"):
            self.iter2_gate = wp.zeros(model.particle_count, dtype=wp.int32, device=self.device)
            # [0] = marked particles in the substep just run (device-side counter)
            self.iter2_gate_stat = wp.zeros(1, dtype=wp.int32, device=self.device)
            # [0] = running maximum of the above over the whole run
            self.iter2_gate_peak = wp.zeros(1, dtype=wp.int32, device=self.device)

        # AERO1: per-triangle aerodynamic drag/lift.  Enabled only when the model
        # actually carries a non-zero drag or lift coefficient
        # (``ModelBuilder.add_triangles(tri_drag=..., tri_lift=...)`` ->
        # ``tri_materials[:, 3:5]``).  With the stock all-zero defaults nothing
        # extra is allocated and ``step()`` hands ``init_step_kernel`` the very
        # same ``state_in.particle_f`` array it did before this feature existed,
        # so the OFF path is unchanged down to the kernel arguments.
        self._aero_enabled = False
        self._f_ext_aero = None
        _tri_mat = getattr(model, "tri_materials", None)
        if _tri_mat is not None and model.tri_count > 0:
            _m = _tri_mat.numpy()
            if _m.shape[1] > 4 and (np.any(_m[:, 3] != 0.0) or np.any(_m[:, 4] != 0.0)):
                self._aero_enabled = True
                self._f_ext_aero = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

        # Drag info
        self.drag_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.drag_index = wp.array([-1], dtype=int, device=self.device)
        self.drag_bary_coord = wp.zeros(1, dtype=wp.vec3, device=self.device)

    # ------------------------------------------------------------ ITER2e
    def _linear_steps(self, _iter: int) -> int:
        """PCG steps for nonlinear iteration ``_iter`` on the non-TP path.

        ``"ramp"`` returns exactly the expression this stack always had, so the
        default path is bit-identical.
        """
        if self.linear_schedule in ("ramp", "stock"):
            return wp.min(_iter + 1, 10)
        if self.linear_schedule == "ramp_it0":
            return self.linear_iterations if _iter == 0 else wp.min(_iter + 1, 10)
        # ITER7 "adaptive": `linear_iterations` is the UPPER bound only; the
        # actual step count is decided on device inside PcgSolver.solve.
        return self.linear_iterations

    # ------------------------------------------------------------ ITER3
    def _linear_steps_tp(self, _iter: int) -> int:
        """PCG steps for nonlinear iteration ``_iter`` on the TP path.

        ``"stock"`` (the default, i.e. nobody named a schedule) returns exactly the
        ``self.linear_iterations`` this path always passed, so the default TP-on
        task is bit-identical.  Naming a schedule now reaches this path too.
        """
        if self.linear_schedule == "stock":
            return self.linear_iterations
        if self.linear_schedule == "ramp":
            return wp.min(_iter + 1, 10)
        if self.linear_schedule == "ramp_it0":
            return self.linear_iterations if _iter == 0 else wp.min(_iter + 1, 10)
        return self.linear_iterations

    # ------------------------------------------------------------- ITER3
    def _cc_update_centroid(self, particle_q) -> None:
        """Per-component centroid, once per substep, from the substep's start state.

        Only the CONDITIONING of the 6x6 depends on the origin -- the rigid-mode
        subspace itself does not -- so a centroid that is one substep stale is
        still an exact Galerkin correction, and it costs one pass instead of two.
        """
        nc = self._translation_component_count
        self._cc_cpartial.zero_()
        wp.launch(
            _cc_centroid_acc_kernel,
            dim=self.model.particle_count,
            inputs=[particle_q, self._translation_particle_component,
                    self.model.particle_flags, _CC_BUCKETS, nc],
            outputs=[self._cc_cpartial],
            device=self.device,
        )
        wp.launch(
            _cc_reduce_kernel,
            dim=nc * _CC_NACC_C,
            inputs=[self._cc_cpartial, _CC_BUCKETS, _CC_NACC_C],
            outputs=[self._cc_ctotal],
            device=self.device,
        )
        wp.launch(
            _cc_centroid_finish_kernel,
            dim=nc,
            inputs=[self._cc_ctotal],
            outputs=[self._cc_centroid],
            device=self.device,
        )

    def _cc_apply(self, dt: float, particle_q, x0) -> None:
        """Add the rigid-mode coarse correction of ``self.rhs`` into ``x0``."""
        nc = self._translation_component_count
        n = self.model.particle_count
        self._cc_partial.zero_()
        if self.coarse_correction == 1:
            wp.launch(
                _cc_moment_t_acc_kernel,
                dim=n,
                inputs=[dt, self.rhs, self._translation_particle_component,
                        self.model.particle_mass, self.model.particle_flags,
                        self._translation_contact_hessian_diags, _CC_BUCKETS, nc],
                outputs=[self._cc_partial],
                device=self.device,
            )
            wp.launch(
                _cc_reduce_kernel,
                dim=nc * _CC_NACC_T,
                inputs=[self._cc_partial, _CC_BUCKETS, _CC_NACC_T],
                outputs=[self._cc_total],
                device=self.device,
            )
            wp.launch(_cc_solve_t_kernel, dim=nc, inputs=[self._cc_total],
                      outputs=[self._cc_t], device=self.device)
            wp.launch(
                _cc_apply_t_kernel,
                dim=n,
                inputs=[self._cc_t, self._translation_particle_component,
                        self.model.particle_flags, nc],
                outputs=[x0],
                device=self.device,
            )
        else:
            wp.launch(
                _cc_moment_r_acc_kernel,
                dim=n,
                inputs=[dt, self.rhs, particle_q, self._cc_centroid,
                        self._translation_particle_component,
                        self.model.particle_mass, self.model.particle_flags,
                        self._translation_contact_hessian_diags, _CC_BUCKETS, nc],
                outputs=[self._cc_partial],
                device=self.device,
            )
            wp.launch(
                _cc_reduce_kernel,
                dim=nc * _CC_NACC_R,
                inputs=[self._cc_partial, _CC_BUCKETS, _CC_NACC_R],
                outputs=[self._cc_total],
                device=self.device,
            )
            wp.launch(_cc_solve_r_kernel, dim=nc, inputs=[self._cc_total],
                      outputs=[self._cc_t, self._cc_w], device=self.device)
            wp.launch(
                _cc_apply_r_kernel,
                dim=n,
                inputs=[self._cc_t, self._cc_w, self._cc_centroid, particle_q,
                        self._translation_particle_component,
                        self.model.particle_flags, nc],
                outputs=[x0],
                device=self.device,
            )

    def iter3_provenance(self) -> str:
        """The one ``[ITER3]`` self-provenance line every run must print."""
        return ("[ITER3] coarse_correction=%d linear_schedule=%s tp=%s "
                "linear_iterations=%d nonlinear_iterations=%d components=%d "
                "tp_reduce=%s pcg_eta=%.4g pcg_fused=%d"
                % (self.coarse_correction, self.linear_schedule,
                   bool(self.enable_translation_preconditioner),
                   int(self.linear_iterations), int(self.nonlinear_iterations),
                   int(self._translation_component_count),
                   self.tp_reduce, self._pcg_eta_active, int(self.pcg_fused)))

    # ------------------------------------------------------------- ITER2
    def _iter2_mark_contact_gate(self, state_in: State, state_out: State,
                                 contacts: Contacts | None = None) -> None:
        """Flag every particle that carries a contact pair in THIS substep.

        Filters the cloth-cloth broad-phase lists ``Collision.frame_begin``
        just built from ``x_prev`` through the force kernels' own narrow-phase
        predicates (the raw lists are proximity lists and would mark the whole
        cloth), and re-runs the tri-SDF gather's AABB test on the same
        positions.  ``state_in.particle_q`` is still ``x_prev`` here.  With no
        collision object nothing is marked and ``free`` == ``all``.
        """
        gate = self.iter2_gate
        gate.zero_()
        # ITER5: the particle-rigid (soft) contacts -- cloth on the TABLE above
        # all -- are a support the ITER2 gate never saw.  Marked FIRST so it is
        # independent of whether a Style3D `collision` object exists at all.
        if contacts is not None and getattr(contacts, "soft_contact_particle", None) is not None:
            wp.launch(
                kernel=iter2_gate_mark_soft_kernel,
                dim=int(contacts.soft_contact_particle.shape[0]),
                inputs=[contacts.soft_contact_count, contacts.soft_contact_particle],
                outputs=[gate],
                device=self.device,
            )
        col = self.collision
        if col is None:
            return
        thickness = 2.0 * float(col.radius)      # same as accumulate_contact_force
        if getattr(col, "stiff_vf", 0.0) > 0.0:
            wp.launch(
                kernel=iter2_gate_mark_vf_kernel,
                dim=self.model.particle_count,
                inputs=[thickness, state_in.particle_q, self.model.tri_indices, col.broad_phase_vf],
                outputs=[gate],
                device=self.device,
            )
        if getattr(col, "stiff_ee", 0.0) > 0.0:
            wp.launch(
                kernel=iter2_gate_mark_ee_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[thickness, state_in.particle_q, self.model.edge_indices, col.broad_phase_ee],
                outputs=[gate],
                device=self.device,
            )
        if getattr(col, "tri_sdf_slot_shape", None) is not None and col.tri_sdf_compliant:
            body_q = (
                state_out.body_q if col.integrate_with_external_rigid_solver else state_in.body_q
            )
            wp.launch(
                kernel=iter2_gate_mark_tri_sdf_kernel,
                dim=col.tri_sdf_slots * int(self.model.tri_count),
                inputs=[
                    state_in.particle_q,
                    self.model.tri_indices,
                    int(self.model.tri_count),
                    col.tri_sdf_slot_shape,
                    self.model.shape_body,
                    self.model.shape_transform,
                    body_q,
                    col.tri_sdf_bp_aabb_lo,
                    col.tri_sdf_bp_aabb_hi,
                    col.tri_sdf_h,
                    col.tri_sdf_bp_slack,
                ],
                outputs=[gate],
                device=self.device,
            )
        # Provenance counter (rule 35: read back from the device, never from the
        # environment).  Device-side only, so it does not break graph capture.
        self.iter2_gate_stat.zero_()
        wp.launch(
            kernel=iter2_gate_count_kernel,
            dim=self.model.particle_count,
            inputs=[gate],
            outputs=[self.iter2_gate_stat],
            device=self.device,
        )
        wp.launch(
            kernel=iter2_gate_peak_kernel,
            dim=1,
            inputs=[self.iter2_gate_stat],
            outputs=[self.iter2_gate_peak],
            device=self.device,
        )

    def iter2_provenance(self) -> str:
        """The one ``[ITER2]`` self-provenance line every run must print.

        ``gated`` is read back from the device counter the last substep wrote,
        never recomputed on the host.
        """
        n = int(self.model.particle_count)
        if self.iter2_gate_stat is None:
            # off / all / accel: there is no gate, so no gated counts to report.
            return f"[ITER2] warm_start={self.inertia_warm_start_mode} particles={n}"
        g = int(self.iter2_gate_stat.numpy()[0])
        pk = int(self.iter2_gate_peak.numpy()[0])
        # ITER5c: `free_v` carries the velocity gate in the same line.
        vg = (" v_gate=%g" % self.inertia_warm_start_v_gate
              if self.inertia_warm_start_mode == "free_v" else "")
        return (f"[ITER2] warm_start={self.inertia_warm_start_mode} "
                f"gated={g}/{n} gated_peak={pk}{vg}")

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float) -> None:
        """Advance the Style3D solver by one time step.

        The solver performs non-linear projective dynamics iterations with
        optional collision handling. During the solve, positions in
        ``state_in`` are updated in-place to the current iterate; the final
        positions and velocities are written to ``state_out``.

        Args:
            state_in: Input :class:`newton.State` (positions updated in-place).
            state_out: Output :class:`newton.State` with the final state.
            control: :class:`newton.Control` input (currently unused).
            contacts: :class:`newton.Contacts` used for collision response.
            dt: Time step in seconds.
        """
        if self.collision is not None:
            self.collision.frame_begin(state_in.particle_q, state_in.particle_qd, dt)
        self._translation_preconditioner_dt = dt
        self._translation_contact_hessian_diags = self._translation_zero_contact_hessian

        # AERO1: fold the aerodynamic force into the external-force array handed to
        # the inertia term.  OFF -> `f_ext` IS `state_in.particle_f` (no copy, no
        # launch, identical kernel arguments).
        f_ext = state_in.particle_f
        if self._aero_enabled:
            self._f_ext_aero.assign(state_in.particle_f)
            wp.launch(
                kernel=eval_aero_force_kernel,
                dim=self.model.tri_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    self.model.tri_indices,
                    self.model.tri_materials,
                ],
                outputs=[self._f_ext_aero],
                device=self.device,
            )
            f_ext = self._f_ext_aero

        wp.launch(
            kernel=init_step_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self.model.gravity,
                self.model.particle_world,
                f_ext,
                state_in.particle_qd,
                state_in.particle_q,
                self.x_prev,
                self.pd_diags,
                self.model.particle_mass,
                self.model.particle_flags,
            ],
            outputs=[
                self.x_inertia,
                self.static_A_diags,
                self.dx,
            ],
            device=self.device,
        )

        # ITER1: hand the first nonlinear iteration the full inertial step as its
        # PCG guess.  Placed right after init_step_kernel, which is what writes
        # both `x_inertia` and the stock `dx = v_prev * dt`.  Nothing between here
        # and the solve reads `dx` (the dragging block writes pd_diags only, and
        # `Collision.linear_iteration_end` is a no-op), and `nonlinear_step_kernel`
        # zeroes `dx` at the end of every iteration, so this only affects iter 0.
        # OFF -> not launched at all.
        if self.inertia_warm_start_mode == "all":
            wp.launch(
                kernel=init_inertia_warm_start_kernel,
                dim=self.model.particle_count,
                inputs=[self.x_inertia, state_in.particle_q],
                outputs=[self.dx],
                device=self.device,
            )
        elif self.inertia_warm_start_mode == "accel_f":
            # ITER2c: only the STARTING POINT changes; `x_inertia` is untouched.
            # `f_contact` was filled at the last nonlinear iteration of the
            # previous substep (see the grab below); it is still zero on the
            # first substep.
            wp.launch(
                kernel=init_accel_f_warm_start_kernel,
                dim=self.model.particle_count,
                inputs=[
                    dt,
                    self.model.gravity,
                    self.model.particle_world,
                    state_in.particle_qd,
                    self.f_contact,
                    self.model.particle_mass,
                    self.model.particle_flags,
                ],
                outputs=[self.dx],
                device=self.device,
            )
        elif self.inertia_warm_start_mode == "accel":
            # ITER2b: only the STARTING POINT changes; `x_inertia` (the inertia
            # term of the energy) is untouched.  The copy below must come after
            # the launch that reads `v_prev2`; both are on the same stream, so
            # the order holds inside a captured graph as well.
            if not self._accel_primed:
                wp.copy(self.v_prev2, state_in.particle_qd)
                self._accel_primed = True
            wp.launch(
                kernel=init_accel_warm_start_kernel,
                dim=self.model.particle_count,
                inputs=[dt, state_in.particle_qd, self.v_prev2, self.model.particle_flags],
                outputs=[self.dx],
                device=self.device,
            )
            wp.copy(self.v_prev2, state_in.particle_qd)
        elif self.inertia_warm_start_mode == "free_v":
            # ITER5c: same gate, plus |v_prev| > v_gate.
            self._iter2_mark_contact_gate(state_in, state_out, contacts)
            wp.launch(
                kernel=init_inertia_warm_start_free_v_kernel,
                dim=self.model.particle_count,
                inputs=[self.inertia_warm_start_v_gate, self.x_inertia,
                        state_in.particle_q, state_in.particle_qd, self.iter2_gate],
                outputs=[self.dx],
                device=self.device,
            )
        elif self.inertia_warm_start_mode == "free":
            # ITER2: mark the contacting particles first, then hand the warm
            # start only to the rest.  Everything below is fixed-dim, device
            # side, no host readback -- the substep stays graph-capturable.
            self._iter2_mark_contact_gate(state_in, state_out, contacts)
            wp.launch(
                kernel=init_inertia_warm_start_free_kernel,
                dim=self.model.particle_count,
                inputs=[self.x_inertia, state_in.particle_q, self.iter2_gate],
                outputs=[self.dx],
                device=self.device,
            )

        # ITER3: one centroid per substep, from the substep-start positions
        # (`state_in.particle_q` is still `x_prev` here).  OFF -> not launched.
        if self.coarse_correction == 6:
            self._cc_update_centroid(state_in.particle_q)

        if self.enable_mouse_dragging:
            wp.launch(
                accumulate_dragging_pd_diag_kernel,
                dim=1,
                inputs=[
                    self.drag_spring_stiff,
                    self.drag_index,
                    self.drag_bary_coord,
                    self.model.tri_indices,
                    self.model.particle_flags,
                ],
                outputs=[self.static_A_diags],
                device=self.device,
            )

        for _iter in range(self.nonlinear_iterations):
            wp.launch(
                init_rhs_kernel,
                dim=self.model.particle_count,
                inputs=[
                    dt,
                    state_in.particle_q,
                    self.x_inertia,
                    self.model.particle_mass,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                eval_stretch_kernel,
                dim=len(self.model.tri_areas),
                inputs=[
                    state_in.particle_q,
                    self.model.tri_areas,
                    self.model.tri_poses,
                    self.model.tri_indices,
                    self.style3d.tri_aniso_ke,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                eval_bend_kernel,
                dim=len(self.style3d.edge_rest_area),
                inputs=[
                    state_in.particle_q,
                    self.style3d.edge_rest_area,
                    self.style3d.edge_bending_cot,
                    self.model.edge_indices,
                    self.model.edge_bending_properties,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            if self.enable_mouse_dragging:
                wp.launch(
                    eval_drag_force_kernel,
                    dim=1,
                    inputs=[
                        self.drag_spring_stiff,
                        self.drag_index,
                        self.drag_pos,
                        self.drag_bary_coord,
                        self.model.tri_indices,
                        state_in.particle_q,
                    ],
                    outputs=[self.rhs],
                    device=self.device,
                )

            # ITER2c: snapshot the rhs right before the contact pass of the
            # LAST nonlinear iteration; the difference across that pass is the
            # contact force alone (inertia/stretch/bend are already in there).
            _grab_fc = (self.inertia_warm_start_mode == "accel_f"
                        and _iter == self.nonlinear_iterations - 1)
            if _grab_fc:
                if self.collision is None:
                    self.f_contact.zero_()
                    _grab_fc = False
                else:
                    wp.copy(self.f_contact, self.rhs)

            if self.collision is not None:
                self.collision.accumulate_contact_force(
                    dt,
                    _iter,
                    state_in,
                    state_out,
                    contacts,
                    self.rhs,
                    self.x_prev,
                    self.static_A_diags,
                )
                self._translation_contact_hessian_diags = self.collision.contact_hessian_diagonal()
                wp.launch(
                    prepare_jacobi_preconditioner_kernel,
                    dim=self.model.particle_count,
                    inputs=[
                        self.static_A_diags,
                        self.collision.contact_hessian_diagonal(),
                        self.model.particle_flags,
                    ],
                    outputs=[self.inv_A_diags],
                    device=self.device,
                )
            else:
                wp.launch(
                    prepare_jacobi_preconditioner_no_contact_hessian_kernel,
                    dim=self.model.particle_count,
                    inputs=[self.static_A_diags],
                    outputs=[self.inv_A_diags],
                    device=self.device,
                )

            if _grab_fc:
                wp.launch(
                    kernel=iter2c_contact_force_delta_kernel,
                    dim=self.model.particle_count,
                    inputs=[self.rhs],
                    outputs=[self.f_contact],
                    device=self.device,
                )

            hessian_multiply = None if self.collision is None else self.collision.hessian_multiply
            # ITER3: the rigid-mode coarse correction goes into the PCG GUESS,
            # here -- after the rhs is complete (inertia + elastic + contact) and
            # before PCG.  OFF -> `_x0` is the very same expression as before and
            # not one kernel is launched.
            if self.coarse_correction:
                if _iter == 0:
                    self._cc_x0.assign(self.dx)
                else:
                    self._cc_x0.zero_()
                self._cc_apply(dt, state_in.particle_q, self._cc_x0)
                _x0 = self._cc_x0
            else:
                _x0 = self.dx if _iter == 0 else None
            if self.enable_translation_preconditioner:
                self.linear_solver.solve(
                    self.pd_non_diags,
                    self.static_A_diags,
                    _x0,
                    self.rhs,
                    self.inv_A_diags,
                    self.dx,
                    # ITER3: the TP path now honours an explicitly named schedule;
                    # "stock" (the default) returns `self.linear_iterations`, which
                    # is the literal value this call always passed.
                    self._linear_steps_tp(_iter),
                    hessian_multiply,
                    self._apply_translation_preconditioner,
                    # ITER7: 0.0 for every schedule but "adaptive".
                    self._pcg_eta_active,
                    # FUSE1: None unless `pcg_fused` (and a fusable TP mode).
                    self._fused_tp_args(),
                )
            else:
                self.linear_solver.solve(
                    self.pd_non_diags,
                    self.static_A_diags,
                    _x0,
                    self.rhs,
                    self.inv_A_diags,
                    self.dx,
                    # ITER1: "ramp" reproduces the historical hard-coded schedule
                    # exactly (and keeps `linear_iterations` unread on this path);
                    # "fixed" honours `linear_iterations` every iteration.
                    # ITER2e: "ramp_it0" gives iteration 0 -- the only one the warm
                    # start's residual lives in, and the one the ramp starves with a
                    # single PCG step -- the full `linear_iterations`, then rejoins
                    # the ramp.  Host-side loop count only; the captured graph stays
                    # a fixed-length unroll either way.
                    self._linear_steps(_iter),
                    hessian_multiply,
                    None,
                    # ITER7: 0.0 for every schedule but "adaptive".
                    self._pcg_eta_active,
                    # FUSE1: None unless `pcg_fused`.
                    self._fused_tp_args(),
                )

            if self.collision is not None:
                self.collision.linear_iteration_end(self.dx)

            wp.launch(
                nonlinear_step_kernel,
                dim=self.model.particle_count,
                inputs=[state_in.particle_q],
                outputs=[state_out.particle_q, self.dx],
                device=self.device,
            )

            if self.collision is not None:
                # PBD-style alternation: project contacts BEFORE the assign, so
                # the next iteration's nonlinear_step_kernel (which reads
                # state_in) starts from the projected positions. Projecting
                # after the assign would silently discard the correction every
                # iteration. No-op unless projection is in interleaved mode.
                self.collision.project_contacts_iteration(
                    particle_q=state_out.particle_q,
                    particle_q_prev=self.x_prev,
                    contacts=contacts,
                    body_q=state_out.body_q if self.collision.integrate_with_external_rigid_solver else state_in.body_q,
                    dt=dt,
                    body_q_prev=state_in.body_q if self.collision.integrate_with_external_rigid_solver else None,
                )

            state_in.particle_q.assign(state_out.particle_q)

        wp.launch(
            kernel=update_velocity,
            dim=self.model.particle_count,
            inputs=[dt, self.vel_damping, self.x_prev, state_out.particle_q],
            outputs=[state_out.particle_qd],
            device=self.device,
        )

        if self.collision is not None:
            self.collision.frame_end(state_out.particle_q, state_out.particle_qd, dt)

    def rebuild_bvh(self, state: State):
        if self.collision is not None:
            self.collision.rebuild_bvh(state.particle_q)

    @staticmethod
    def _build_particle_components(model: Model) -> tuple[int, np.ndarray]:
        parent = list(range(model.particle_count))
        rank = [0] * model.particle_count
        particle_world = model.particle_world.numpy() if model.particle_world is not None else None

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            if a < 0 or b < 0 or a >= model.particle_count or b >= model.particle_count:
                return
            if particle_world is not None and particle_world[a] != particle_world[b]:
                return
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        if model.tri_indices is not None:
            for tri in model.tri_indices.numpy():
                a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
                union(a, b)
                union(b, c)
                union(c, a)

        component_map: dict[int, int] = {}
        particle_component = np.empty(model.particle_count, dtype=np.int32)
        for particle in range(model.particle_count):
            root = find(particle)
            component = component_map.setdefault(root, len(component_map))
            particle_component[particle] = component
        return max(1, len(component_map)), particle_component

    def _fused_tp_args(self):
        """FUSE1: the struct the fused PCG kernels need, or ``None`` when the
        fused path must fall back to the stock loop.

        Returns ``None`` for ``tp_reduce="bucket"`` (its two-level reduction is
        not fused) and whenever ``pcg_fused`` is off.  With TP off it returns an
        ``enabled=0`` struct, which still takes the fused path.
        """
        if not self.pcg_fused:
            return None
        tp = self._fused_tp
        if not self.enable_translation_preconditioner:
            tp.enabled = 0
            return tp
        if self.tp_reduce != "legacy":
            return None
        tp.enabled = 1
        tp.dt = self._translation_preconditioner_dt
        tp.contact_hessian_diags = self._translation_contact_hessian_diags
        # The in-loop re-zero is done by `fused_mat_vec_pTAp_kernel`; only the
        # prologue's accumulate needs a memset here (2 per solve, not per step).
        self._translation_coarse_rhs.zero_()
        self._translation_coarse_diag.zero_()
        return tp

    def _apply_translation_preconditioner(self, residual: wp.array[wp.vec3], z: wp.array[wp.vec3]) -> None:
        # ITER7: "bucket" replaces the single-slot atomic accumulation with the
        # two-level bucketed reduction (same accumulators, different summation
        # ORDER -- so not bit-identical; default stays "legacy").
        if self.tp_reduce == "bucket":
            self._apply_translation_preconditioner_bucket(residual, z)
            return
        self._translation_coarse_rhs.zero_()
        self._translation_coarse_diag.zero_()
        wp.launch(
            _accumulate_translation_preconditioner_kernel,
            dim=self.model.particle_count,
            inputs=[
                self._translation_preconditioner_dt,
                residual,
                self._translation_particle_component,
                self.model.particle_mass,
                self.model.particle_flags,
                self._translation_contact_hessian_diags,
            ],
            outputs=[
                self._translation_coarse_rhs,
                self._translation_coarse_diag,
            ],
            device=self.device,
        )
        wp.launch(
            _apply_translation_preconditioner_kernel,
            dim=self.model.particle_count,
            inputs=[
                self._translation_coarse_rhs,
                self._translation_coarse_diag,
                self._translation_particle_component,
                self.model.particle_flags,
            ],
            outputs=[z],
            device=self.device,
        )

    # ------------------------------------------------------------- ITER7
    def _apply_translation_preconditioner_bucket(
        self, residual: wp.array[wp.vec3], z: wp.array[wp.vec3]
    ) -> None:
        """Bucketed two-level reduction (`tp_reduce="bucket"`).

        Same accumulators as the legacy kernel (sum r, sum a), but each particle
        adds into `tid % _TP_BUCKETS` of its component first, so no single
        address takes every atomic.  The second level reuses `_cc_reduce_kernel`.
        """
        n_comp = self._translation_coarse_diag.shape[0]
        if self._tp_partial is None:
            self._tp_partial = wp.zeros(
                n_comp * _TP_BUCKETS * _TP_NACC, dtype=float, device=self.device
            )
            self._tp_total = wp.zeros(n_comp * _TP_NACC, dtype=float, device=self.device)
        self._tp_partial.zero_()
        wp.launch(
            _tp_bucket_acc_kernel,
            dim=self.model.particle_count,
            inputs=[
                self._translation_preconditioner_dt,
                residual,
                self._translation_particle_component,
                self.model.particle_mass,
                self.model.particle_flags,
                self._translation_contact_hessian_diags,
                _TP_BUCKETS,
                n_comp,
            ],
            outputs=[self._tp_partial],
            device=self.device,
        )
        wp.launch(
            _cc_reduce_kernel,
            dim=n_comp * _TP_NACC,
            inputs=[self._tp_partial, _TP_BUCKETS, _TP_NACC],
            outputs=[self._tp_total],
            device=self.device,
        )
        wp.launch(
            _tp_bucket_finish_kernel,
            dim=n_comp,
            inputs=[self._tp_total],
            outputs=[self._translation_coarse_rhs, self._translation_coarse_diag],
            device=self.device,
        )
        wp.launch(
            _apply_translation_preconditioner_kernel,
            dim=self.model.particle_count,
            inputs=[
                self._translation_coarse_rhs,
                self._translation_coarse_diag,
                self._translation_particle_component,
                self.model.particle_flags,
            ],
            outputs=[z],
            device=self.device,
        )

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Declare Style3D custom attributes under the ``style3d`` namespace.

        See Also:
            :ref:`custom_attributes` for the custom attribute system overview.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tri_aniso_ke",
                frequency=AttributeFrequency.TRIANGLE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="edge_rest_area",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="edge_bending_cot",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec4,
                default=wp.vec4(0.0, 0.0, 0.0, 0.0),
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="aniso_ke",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="style3d",
            )
        )

    def _precompute(self, model: Model):
        with wp.ScopedTimer("SolverStyle3D::precompute()"):
            if (
                not hasattr(model, "style3d")
                or not hasattr(model.style3d, "tri_aniso_ke")
                or not hasattr(model.style3d, "edge_rest_area")
                or not hasattr(model.style3d, "edge_bending_cot")
            ):
                raise AttributeError(
                    "Style3D custom attributes are missing from the model. "
                    "Call SolverStyle3D.register_custom_attributes() before building the model."
                )

            pd_matrix_builder = PDMatrixBuilder(model.particle_count)
            tri_indices = model.tri_indices.numpy().tolist()
            tri_poses = model.tri_poses.numpy().tolist()
            tri_areas = model.tri_areas.numpy().tolist()
            edge_indices = model.edge_indices.numpy().tolist()
            edge_bending_properties = model.edge_bending_properties.numpy().tolist()
            tri_aniso_ke = model.style3d.tri_aniso_ke.numpy().tolist()
            edge_rest_area = model.style3d.edge_rest_area.numpy().tolist()
            edge_bending_cot = model.style3d.edge_bending_cot.numpy().tolist()

            pd_matrix_builder.add_stretch_constraints(tri_indices, tri_poses, tri_aniso_ke, tri_areas)
            pd_matrix_builder.add_bend_constraints(
                edge_indices,
                edge_bending_properties,
                edge_rest_area,
                edge_bending_cot,
            )
            self.pd_diags, self.pd_non_diags.num_nz, self.pd_non_diags.nz_ell = pd_matrix_builder.finalize(self.device)

    def _update_drag_info(self, index: int, pos: wp.vec3, bary_coord: wp.vec3):
        """Should be invoked when state changed."""
        # print([index, pos, bary_coord])
        self.drag_bary_coord.fill_(bary_coord)
        self.drag_index.fill_(index)
        self.drag_pos.fill_(pos)
