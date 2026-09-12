# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import warp as wp

from ...geometry import ParticleFlags


@wp.struct
class NonZeroEntry:
    """Represents a non-zero entry in a sparse matrix.
    This structure stores the column index and corresponding value in a packed format, which provides
    better cache locality for sequential access patterns.
    """

    column_index: int
    value: float


@wp.struct
class SparseMatrixELL:
    """Represents a sparse matrix in ELLPACK (ELL) format."""

    num_nz: wp.array[int]  # Non-zeros count per column
    nz_ell: wp.array2d[NonZeroEntry]  # Padded ELL storage [row-major, fixed-height]


@wp.func
def ell_mat_vec_mul(
    num_nz: wp.array[int],
    nz_ell: wp.array2d[NonZeroEntry],
    x: wp.array[wp.vec3],
    tid: int,
):
    Mx = wp.vec3(0.0)
    for k in range(num_nz[tid]):
        nz_entry = nz_ell[k, tid]
        Mx += x[nz_entry.column_index] * nz_entry.value
    return Mx


@wp.kernel
def eval_residual_kernel(
    A_non_diag: SparseMatrixELL,
    A_diag: wp.array[Any],
    x: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    # outputs
    r: wp.array[wp.vec3],
):
    tid = wp.tid()
    Ax = A_diag[tid] * x[tid]
    Ax += ell_mat_vec_mul(A_non_diag.num_nz, A_non_diag.nz_ell, x, tid)
    r[tid] = b[tid] - Ax


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(eval_residual_kernel, {"A_diag": wp.array[wp.float32]})
wp.overload(eval_residual_kernel, {"A_diag": wp.array[wp.mat33]})


@wp.kernel
def eval_residual_kernel_with_additional_Ax(
    A_non_diag: SparseMatrixELL,
    A_diag: wp.array[Any],
    x: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    additional_Ax: wp.array[wp.vec3],
    # outputs
    r: wp.array[wp.vec3],
):
    tid = wp.tid()
    Ax = A_diag[tid] * x[tid] + additional_Ax[tid]
    Ax += ell_mat_vec_mul(A_non_diag.num_nz, A_non_diag.nz_ell, x, tid)
    r[tid] = b[tid] - Ax


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(eval_residual_kernel_with_additional_Ax, {"A_diag": wp.array[wp.float32]})
wp.overload(eval_residual_kernel_with_additional_Ax, {"A_diag": wp.array[wp.mat33]})


@wp.kernel
def array_mul_kernel(
    a: wp.array[Any],
    b: wp.array[wp.vec3],
    # outputs
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    out[tid] = a[tid] * b[tid]


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(array_mul_kernel, {"a": wp.array[wp.float32]})
wp.overload(array_mul_kernel, {"a": wp.array[wp.mat33]})


@wp.kernel
def ell_mat_vec_mul_kernel(
    M_non_diag: SparseMatrixELL,
    M_diag: wp.array[Any],
    x: wp.array[wp.vec3],
    # outputs
    Mx: wp.array[wp.vec3],
):
    tid = wp.tid()
    Mx[tid] = (M_diag[tid] * x[tid]) + ell_mat_vec_mul(M_non_diag.num_nz, M_non_diag.nz_ell, x, tid)


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(ell_mat_vec_mul_kernel, {"M_diag": wp.array[wp.float32]})
wp.overload(ell_mat_vec_mul_kernel, {"M_diag": wp.array[wp.mat33]})


@wp.kernel
def ell_mat_vec_mul_add_kernel(
    M_non_diag: SparseMatrixELL,
    M_diag: wp.array[Any],
    x: wp.array[wp.vec3],
    additional_Mx: wp.array[wp.vec3],
    # outputs
    Mx: wp.array[wp.vec3],
):
    tid = wp.tid()
    result = (M_diag[tid] * x[tid]) + additional_Mx[tid]
    result += ell_mat_vec_mul(M_non_diag.num_nz, M_non_diag.nz_ell, x, tid)
    Mx[tid] = result


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(ell_mat_vec_mul_add_kernel, {"M_diag": wp.array[wp.float32]})
wp.overload(ell_mat_vec_mul_add_kernel, {"M_diag": wp.array[wp.mat33]})


@wp.kernel
def update_cg_direction_kernel(
    iter: int,
    z: wp.array[wp.vec3],
    rTz: wp.array[float],
    p_prev: wp.array[wp.vec3],
    # outputs
    p: wp.array[wp.vec3],
):
    # p = r + (rz_new / rz_old) * p;
    i = wp.tid()
    new_p = z[i]
    if iter > 0:
        num = rTz[iter]
        denom = rTz[iter - 1]
        beta = wp.float32(0.0)
        if (wp.abs(denom) > 1.0e-30) and (not wp.isnan(denom)) and (not wp.isnan(num)):
            beta = num / denom
        new_p += beta * p_prev[i]
    p[i] = new_p


@wp.kernel
def step_cg_kernel(
    iter: int,
    rTz: wp.array[float],
    pTAp: wp.array[float],
    p: wp.array[wp.vec3],
    Ap: wp.array[wp.vec3],
    # outputs
    x: wp.array[wp.vec3],
    r: wp.array[wp.vec3],
):
    i = wp.tid()
    num = rTz[iter]
    denom = pTAp[iter]
    alpha = wp.float32(0.0)
    if (wp.abs(denom) > 1.0e-30) and (not wp.isnan(denom)) and (not wp.isnan(num)):
        alpha = num / denom
    r[i] = r[i] - alpha * Ap[i]
    x[i] = x[i] + alpha * p[i]


# ------------------------------------------------------------------- ITER7
# Adaptive PCG step count (`linear_schedule="adaptive"`).
#
# Stop when the preconditioned residual has dropped by eta:  with z = M^-1 r,
# `rTz[k]` IS ||r_k||^2 in the M^-1 norm, so the standard relative test is
#
#       sqrt(rTz[k] / rTz[0]) <= eta      <=>      rTz[k] <= eta^2 * rTz[0]
#
# `rTz` is already computed by step3 every iteration, so the test costs one
# 1-thread kernel and NO host readback.  The Python loop still runs the full
# `linear_iterations` trips (the captured CUDA graph stays a fixed-length
# unroll); the iterations after the stop are turned into no-ops on device by
# `stop_flag`, which freezes p, x and r.  Lower bound is 1 step: iteration 0
# always executes, the test only runs for iter >= 1.
#
# NOTE(cost): because the unroll is fixed, `adaptive` does NOT save kernel
# launches under CUDA graph capture -- it changes the ITERATE, not the launch
# count.  What it buys is the step-count distribution (`pcg_steps_hist`) and a
# solution that stops at a residual target instead of a fixed trip count.
@wp.kernel
def pcg_check_stop_kernel(
    iter: int,
    eta2: float,
    rTz: wp.array[float],
    # outputs
    stop_flag: wp.array[wp.int32],
    steps_taken: wp.array[wp.int32],
):
    if iter == 0:
        stop_flag[0] = 0
        steps_taken[0] = 1          # lower bound: iteration 0 always runs
        return
    if stop_flag[0] != 0:
        return
    r0 = rTz[0]
    rk = rTz[iter]
    if (not wp.isnan(rk)) and (not wp.isnan(r0)) and (r0 > 0.0) and (rk <= eta2 * r0):
        stop_flag[0] = 1            # steps_taken stays at `iter`
    else:
        steps_taken[0] = iter + 1


@wp.kernel
def pcg_record_steps_kernel(
    steps_taken: wp.array[wp.int32],
    # outputs
    hist: wp.array[wp.int32],
):
    k = steps_taken[0]
    if k >= 0 and k < hist.shape[0]:
        wp.atomic_add(hist, k, 1)


@wp.kernel
def update_cg_direction_gated_kernel(
    iter: int,
    z: wp.array[wp.vec3],
    rTz: wp.array[float],
    p_prev: wp.array[wp.vec3],
    stop_flag: wp.array[wp.int32],
    # outputs
    p: wp.array[wp.vec3],
):
    # Byte-for-byte `update_cg_direction_kernel` plus the ITER7 stop gate.
    if stop_flag[0] != 0:
        return
    i = wp.tid()
    new_p = z[i]
    if iter > 0:
        num = rTz[iter]
        denom = rTz[iter - 1]
        beta = wp.float32(0.0)
        if (wp.abs(denom) > 1.0e-30) and (not wp.isnan(denom)) and (not wp.isnan(num)):
            beta = num / denom
        new_p += beta * p_prev[i]
    p[i] = new_p


@wp.kernel
def step_cg_gated_kernel(
    iter: int,
    rTz: wp.array[float],
    pTAp: wp.array[float],
    p: wp.array[wp.vec3],
    Ap: wp.array[wp.vec3],
    stop_flag: wp.array[wp.int32],
    # outputs
    x: wp.array[wp.vec3],
    r: wp.array[wp.vec3],
):
    # Byte-for-byte `step_cg_kernel` plus the ITER7 stop gate.
    if stop_flag[0] != 0:
        return
    i = wp.tid()
    num = rTz[iter]
    denom = pTAp[iter]
    alpha = wp.float32(0.0)
    if (wp.abs(denom) > 1.0e-30) and (not wp.isnan(denom)) and (not wp.isnan(num)):
        alpha = num / denom
    r[i] = r[i] - alpha * Ap[i]
    x[i] = x[i] + alpha * p[i]


# ------------------------------------------------------------------- FUSE1
# Fused PCG inner loop (`pcg_fused=True`, default OFF -> stock path untouched).
#
# The stock loop launches, per PCG step:
#     array_mul (z)                             [+ TP: 2 memsets + accumulate + apply]
#     array_inner (rTz) -> update_cg_direction (p) -> ell_mat_vec (Ap)
#     array_inner (pTAp) -> step_cg (x, r)
# = 6 launches, 10 with the translation preconditioner.  All of them are pure
# element-wise passes over `dim` particles plus two whole-array dot products;
# at ~5 k particles each launch is latency, not work.
#
# A PCG step has exactly three GLOBAL dependencies, so three kernels is the
# floor:
#     (a) `p` complete before `Ap = A p`   (the ELL row reads neighbours)
#     (b) `pTAp` complete before alpha     (reduction)
#     (c) `rTz` complete before beta       (reduction)
# Everything else folds into those three:
#     K1  fused_update_p_kernel      = update_cg_direction + rTz bucket sum -> p
#     K2  fused_mat_vec_pTAp_kernel  = ell_mat_vec + pTAp                   -> Ap, pTAp
#     K3  fused_step_z_kernel        = step_cg + array_mul + rTz            -> x, r, z, rTz
#
# With the translation preconditioner ON one more point appears -- the coarse
# sums must be complete before the correction can be applied, and rTz is only
# defined on the CORRECTED z:
#     K3' fused_step_z_kernel   = step_cg + array_mul + TP accumulate
#     K4  fused_tp_apply_kernel = TP apply + rTz
# so TP costs ZERO extra launches per step on the fused path (4 vs 3), where
# the stock path pays four (2 memsets + accumulate + apply).  The coarse
# accumulators are re-zeroed by K2 (its threads with tid < n_comp), one kernel
# ahead of the K3' that refills them, so no memset is needed inside the loop.
#
# ------------------------------------------------------------------- FUSE2
# THE REDUCTIONS ARE TWO-LEVEL (`pcg_fused_buckets`, default 32).
#
# FUSE1 replaced `array_inner` with a single-address `wp.atomic_add`.  That is
# free at 5 k particles but serialises with N: measured on one H20, a
# single-address atomic dot costs 13.7 us at 4.7 k, 18.1 at 9.4 k, 37.2 at
# 20.3 k and 177 at 100 k, while `array_inner`'s tree reduction is a flat
# 10 us.  So FUSE1 won on silk/tshirt and LOST 4-8 % on the 20 k bag.
#
# FUSE2 keeps the three-kernel shape and kills the contention instead:
#   level 1  each thread atomically adds into bucket `tid % FUSE_BUCKETS` of a
#            [iteration, nb] scratch row -> contention drops by nb;
#   level 2  the CONSUMER kernel sums those nb floats itself, redundantly, in
#            every thread (`_fused_bucket_sum`).  nb broadcast loads out of L1
#            are far cheaper than an extra kernel launch, so the launch count
#            is unchanged -- still 3 per step (4 with TP).
# `pcg_fused_buckets=1` reproduces FUSE1's single-address behaviour exactly and
# is kept as the A/B arm.
#
# The stock `update_cg_direction_kernel` is NOT touched: the bucket sum lives
# in a new `fused_update_p_kernel` (K1 above), so the unfused path keeps its
# kernels byte-for-byte.  ITER7's `adaptive` gate kernels are not touched
# either -- `adaptive` and `pcg_fused` do not combine (see `PcgSolver.solve`),
# so the gate keeps reading the 1-D `rTz` that the unfused loop still writes.
#
# NUMERICS.  Every element-wise expression below is copied verbatim, in the
# same order, from the kernel it replaces, so the per-particle arithmetic is
# bit-identical.  Only the two REDUCTIONS differ: the summation order changes
# (atomics within a bucket race; the bucket-to-scalar sum is a fixed 0..nb-1
# loop).  So fused ON is neither bit-equal to fused OFF nor bit-reproducible
# run to run on GPU.  Hence the flag defaults to OFF.
#
# Because the slots are accumulated instead of assigned, `solve()` zeroes the
# two bucket scratch arrays once per call (2 memsets against ~30 launches
# saved).


# Level-1 bucket count.  COMPILE-TIME on purpose, twice over:
#   * `tid % FUSE_BUCKETS` becomes a bitwise AND instead of an integer divide,
#     and the kernels lose an argument -- measured, a runtime bucket count cost
#     +0.32 ms/substep on silk_4k (~8 us per PCG step);
#   * `_fused_bucket_sum` unrolls (a dynamic 32-trip sum costs +2.15 us over
#     the element-wise floor, an unrolled one +0.93 us, and two unrolled sums
#     then cost the same as one).
# 8 is the measured optimum on one H20 (pure-cloth bench, `prod` arm, 500
# substeps x 3): bag 20 k particles  nb=1 25.98 / nb=4 20.52 / nb=8 19.00 /
# nb=32 20.02 ms per substep;  silk 4.7 k is flat (8.52-8.64) across all four,
# so nothing is lost at the small end.  Changing it is a recompile, not a knob.
FUSE_BUCKETS = wp.constant(8)


@wp.func
def _fused_bucket_sum(buckets: wp.array2d(dtype=float), row: int):
    """Level-2 reduction: sum one row of the bucket scratch, in a fixed order.

    Run redundantly by every thread of the consumer kernel -- 32 broadcast
    loads out of L1, which is what buys back the kernel launch this would
    otherwise cost.
    """
    s = float(0.0)
    for k in range(FUSE_BUCKETS):
        s += buckets[row, k]
    return s


@wp.struct
class FusedTranslationPrecond:
    """Everything the legacy translation preconditioner needs, as one arg.

    `enabled == 0` means "TP off": the array members are then never touched
    (they are still bound, so the struct marshals the same way either way).
    """

    enabled: int
    n_comp: int
    dt: float
    particle_component: wp.array[wp.int32]
    particle_masses: wp.array[wp.float32]
    particle_flags: wp.array[wp.int32]
    contact_hessian_diags: wp.array[wp.mat33]
    coarse_rhs: wp.array[wp.vec3]
    coarse_diag: wp.array[wp.vec3]


@wp.kernel
def fused_z_kernel(
    iter: int,
    inv_M: wp.array[Any],
    r: wp.array[wp.vec3],
    tp: FusedTranslationPrecond,
    # outputs
    z: wp.array[wp.vec3],
    rTz_b: wp.array2d(dtype=float),
):
    """PCG prologue: `array_mul_kernel` fused with either the rTz reduction
    (TP off) or the TP coarse accumulation (TP on -- then `fused_tp_apply_kernel`
    finishes rTz on the corrected z)."""
    tid = wp.tid()
    ri = r[tid]
    zi = inv_M[tid] * ri
    z[tid] = zi
    if tp.enabled != 0:
        # verbatim `_accumulate_translation_preconditioner_kernel`
        if tp.particle_flags[tid] & ParticleFlags.ACTIVE:
            comp_idx = wp.min(wp.max(tp.particle_component[tid], 0), tp.n_comp - 1)
            mass_diag = tp.particle_masses[tid] / (tp.dt * tp.dt)
            contact_hess = tp.contact_hessian_diags[tid]
            wp.atomic_add(tp.coarse_rhs, comp_idx, ri)
            wp.atomic_add(
                tp.coarse_diag,
                comp_idx,
                wp.vec3(
                    mass_diag + contact_hess[0, 0],
                    mass_diag + contact_hess[1, 1],
                    mass_diag + contact_hess[2, 2],
                ),
            )
    else:
        wp.atomic_add(rTz_b, iter, tid % FUSE_BUCKETS, wp.dot(ri, zi))


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(fused_z_kernel, {"inv_M": wp.array[wp.float32]})
wp.overload(fused_z_kernel, {"inv_M": wp.array[wp.mat33]})


@wp.kernel
def fused_tp_apply_kernel(
    iter: int,
    tp: FusedTranslationPrecond,
    r: wp.array[wp.vec3],
    # outputs
    z: wp.array[wp.vec3],
    rTz_b: wp.array2d(dtype=float),
):
    """`_apply_translation_preconditioner_kernel` fused with the rTz reduction.

    rTz is the inner product over ALL particles (the stock `array_inner` does
    not look at the ACTIVE flag), so the dot runs unconditionally on the final
    value of z -- corrected for active particles, untouched for the rest.
    """
    tid = wp.tid()
    comp_idx = wp.min(wp.max(tp.particle_component[tid], 0), tp.n_comp - 1)
    denom = tp.coarse_diag[comp_idx]
    rhs = tp.coarse_rhs[comp_idx]
    zi = z[tid]
    if tp.particle_flags[tid] & ParticleFlags.ACTIVE:
        correction = wp.vec3(0.0)
        if denom[0] > 0.0:
            correction[0] = rhs[0] / denom[0]
        if denom[1] > 0.0:
            correction[1] = rhs[1] / denom[1]
        if denom[2] > 0.0:
            correction[2] = rhs[2] / denom[2]
        zi = zi + correction
        z[tid] = zi
    wp.atomic_add(rTz_b, iter, tid % FUSE_BUCKETS, wp.dot(r[tid], zi))


@wp.kernel
def fused_update_p_kernel(
    iter: int,
    z: wp.array[wp.vec3],
    rTz_b: wp.array2d(dtype=float),
    p_prev: wp.array[wp.vec3],
    # outputs
    p: wp.array[wp.vec3],
):
    """`update_cg_direction_kernel` with beta taken from the bucket scratch.

    The stock kernel is left untouched for the unfused path; this is the only
    difference between the two (`num`/`denom` come from `_fused_bucket_sum`
    instead of the 1-D `rTz`).
    """
    i = wp.tid()
    new_p = z[i]
    if iter > 0:
        num = _fused_bucket_sum(rTz_b, iter)
        denom = _fused_bucket_sum(rTz_b, iter - 1)
        beta = wp.float32(0.0)
        if (wp.abs(denom) > 1.0e-30) and (not wp.isnan(denom)) and (not wp.isnan(num)):
            beta = num / denom
        new_p += beta * p_prev[i]
    p[i] = new_p


@wp.kernel
def fused_mat_vec_pTAp_kernel(
    iter: int,
    M_non_diag: SparseMatrixELL,
    M_diag: wp.array[Any],
    x: wp.array[wp.vec3],
    additional_Mx: wp.array[wp.vec3],
    use_additional: int,
    tp: FusedTranslationPrecond,
    # outputs
    Mx: wp.array[wp.vec3],
    pTAp_b: wp.array2d(dtype=float),
):
    """`ell_mat_vec_mul[_add]_kernel` + the pTAp reduction, and (TP on) the
    re-zeroing of the coarse accumulators for the NEXT step's accumulate."""
    tid = wp.tid()
    xi = x[tid]
    # `result` is declared before the branch (warp only merges mutations of
    # variables that already exist).  Both association orders below are
    # verbatim: (d*x + additional) + ell  /  (d*x) + ell.
    result = M_diag[tid] * xi
    if use_additional != 0:
        result = result + additional_Mx[tid]
        result = result + ell_mat_vec_mul(M_non_diag.num_nz, M_non_diag.nz_ell, x, tid)
    else:
        result = result + ell_mat_vec_mul(M_non_diag.num_nz, M_non_diag.nz_ell, x, tid)
    Mx[tid] = result
    wp.atomic_add(pTAp_b, iter, tid % FUSE_BUCKETS, wp.dot(xi, result))
    if tp.enabled != 0:
        if tid < tp.n_comp:
            tp.coarse_rhs[tid] = wp.vec3(0.0)
            tp.coarse_diag[tid] = wp.vec3(0.0)


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(fused_mat_vec_pTAp_kernel, {"M_diag": wp.array[wp.float32]})
wp.overload(fused_mat_vec_pTAp_kernel, {"M_diag": wp.array[wp.mat33]})


@wp.kernel
def fused_step_z_kernel(
    iter: int,
    do_next: int,
    rTz_b: wp.array2d(dtype=float),
    pTAp_b: wp.array2d(dtype=float),
    p: wp.array[wp.vec3],
    Ap: wp.array[wp.vec3],
    inv_M: wp.array[Any],
    tp: FusedTranslationPrecond,
    # outputs
    x: wp.array[wp.vec3],
    r: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
):
    """`step_cg_kernel` + `array_mul_kernel` for the NEXT step's z, fused with
    either that step's rTz reduction (TP off) or the TP coarse accumulation.

    `do_next == 0` on the LAST PCG step: nobody reads z / rTz again, so only
    x and r are updated (and rTz[iter+1] is never indexed out of range).
    """
    i = wp.tid()
    num = _fused_bucket_sum(rTz_b, iter)
    denom = _fused_bucket_sum(pTAp_b, iter)
    alpha = wp.float32(0.0)
    if (wp.abs(denom) > 1.0e-30) and (not wp.isnan(denom)) and (not wp.isnan(num)):
        alpha = num / denom
    ri = r[i] - alpha * Ap[i]
    r[i] = ri
    x[i] = x[i] + alpha * p[i]
    if do_next == 0:
        return
    zi = inv_M[i] * ri
    z[i] = zi
    if tp.enabled != 0:
        if tp.particle_flags[i] & ParticleFlags.ACTIVE:
            comp_idx = wp.min(wp.max(tp.particle_component[i], 0), tp.n_comp - 1)
            mass_diag = tp.particle_masses[i] / (tp.dt * tp.dt)
            contact_hess = tp.contact_hessian_diags[i]
            wp.atomic_add(tp.coarse_rhs, comp_idx, ri)
            wp.atomic_add(
                tp.coarse_diag,
                comp_idx,
                wp.vec3(
                    mass_diag + contact_hess[0, 0],
                    mass_diag + contact_hess[1, 1],
                    mass_diag + contact_hess[2, 2],
                ),
            )
    else:
        wp.atomic_add(rTz_b, iter + 1, i % FUSE_BUCKETS, wp.dot(ri, zi))


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(fused_step_z_kernel, {"inv_M": wp.array[wp.float32]})
wp.overload(fused_step_z_kernel, {"inv_M": wp.array[wp.mat33]})


@wp.kernel
def generate_test_data_kernel(
    dim: int,
    diag_term: float,
    A_non_diag: SparseMatrixELL,
    A_diag: wp.array[Any],
    b: wp.array[wp.vec3],
    x0: wp.array[wp.vec3],
):
    tid = wp.tid()

    t = wp.float32(tid)
    b[tid] = wp.vec3(wp.sin(t * 0.123), wp.cos(t * 0.456), wp.sin(t * 0.789))
    x0[tid] = wp.vec3(wp.cos(t * 0.123), wp.tan(t * 0.456), wp.cos(t * 0.789))

    A_diag[tid] = diag_term

    if tid == 0:
        A_non_diag.num_nz[tid] = 1
        A_non_diag.nz_ell[0, tid].value = -1.0
        A_non_diag.nz_ell[0, tid].column_index = 1
    elif tid == dim - 1:
        A_non_diag.num_nz[tid] = 1
        A_non_diag.nz_ell[0, tid].value = -1.0
        A_non_diag.nz_ell[0, tid].column_index = dim - 2
    else:
        A_non_diag.num_nz[tid] = 2
        A_non_diag.nz_ell[0, tid].value = -1.0
        A_non_diag.nz_ell[0, tid].column_index = tid + 1
        A_non_diag.nz_ell[1, tid].value = -1.0
        A_non_diag.nz_ell[1, tid].column_index = tid - 1


def array_inner(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    out_ptr: wp.uint64,
):
    from warp._src.context import runtime  # noqa: PLC0415

    if a.device.is_cpu:
        func = runtime.core.wp_array_inner_float_host
    else:
        func = runtime.core.wp_array_inner_float_device

    func(
        a.ptr,
        b.ptr,
        out_ptr,
        len(a),
        wp.types.type_size_in_bytes(a.dtype),
        wp.types.type_size_in_bytes(b.dtype),
        wp.types.type_size(a.dtype),
    )


class PcgSolver:
    """A Customized PCG implementation for efficient cloth simulation

    Ref: https://en.wikipedia.org/wiki/Conjugate_gradient_method

    Sparse Matrix Storages:
        Part-1: (static)
            1. Non-diagonals: SparseMatrixELL
            2. Diagonals: wp.array(dtype = mat3x3)
        Part-2: (dynamic)
            1. Preconditioner: wp.array(wp.mat3x3)
            2. Matrix-free Ax: wp.array(dtype = wp.vec3)
            3. Matrix-free diagonals: wp.array(wp.mat3x3)
    """

    def __init__(self, dim: int, device, maxIter: int = 999, fused: bool = False):
        self.dim = dim  # pre-allocation
        self.device = device
        self.maxIter = maxIter
        # FUSE1: fuse the per-step element-wise kernels (default OFF).
        self.fused = bool(fused)
        self._fused_tp_off = None  # lazily built `enabled=0` struct
        # FUSE2: level-1 bucket scratch for the two dot products.
        if self.fused:
            self.rTz_b = wp.zeros((maxIter, int(FUSE_BUCKETS)), dtype=float, device=device)
            self.pTAp_b = wp.zeros((maxIter, int(FUSE_BUCKETS)), dtype=float, device=device)
        else:
            self.rTz_b = None
            self.pTAp_b = None
        self.r = wp.array(shape=dim, dtype=wp.vec3, device=device)
        self.z = wp.array(shape=dim, dtype=wp.vec3, device=device)
        self.p = wp.array(shape=dim, dtype=wp.vec3, device=device)
        self.Ap = wp.array(shape=dim, dtype=wp.vec3, device=device)
        self.pTAp = wp.array(shape=maxIter, dtype=float, device=device)
        self.rTz = wp.array(shape=maxIter, dtype=float, device=device)
        # ------------------------------------------------------------ ITER7
        # `adaptive` state.  Allocated unconditionally (a few ints) but only
        # touched when `solve(..., eta=...)` is called with eta > 0, so the
        # default path launches not one extra kernel.
        self._stop_flag = wp.zeros(1, dtype=wp.int32, device=device)
        self._steps_taken = wp.zeros(1, dtype=wp.int32, device=device)
        # pcg_steps_hist[k] = how many PCG solves executed exactly k steps.
        self.pcg_steps_hist = wp.zeros(maxIter + 1, dtype=wp.int32, device=device)

    def step1_update_r(
        self,
        A_non_diag: SparseMatrixELL,
        A_diag: wp.array[Any],
        b: wp.array[wp.vec3],
        x: wp.array[wp.vec3] = None,  # Pass `None` if x[:] == 0.0
        additional_Ax: wp.array[wp.vec3] = None,  # Pass `None` if additional_Ax[:] == 0.0
    ):
        """Update residual: r = b - A * x"""
        if x is None:
            self.r.assign(b)
        elif additional_Ax is None:
            wp.launch(
                eval_residual_kernel,
                dim=self.dim,
                inputs=[A_non_diag, A_diag, x, b],
                outputs=[self.r],
                device=self.device,
            )
        else:
            wp.launch(
                eval_residual_kernel_with_additional_Ax,
                dim=self.dim,
                inputs=[A_non_diag, A_diag, x, b, additional_Ax],
                outputs=[self.r],
                device=self.device,
            )

    def step2_update_z(self, inv_M: wp.array[Any]):
        wp.launch(array_mul_kernel, dim=self.dim, inputs=[inv_M, self.r], outputs=[self.z], device=self.device)

    def step3_update_rTz(self, iter: int):
        array_inner(self.r, self.z, self.rTz.ptr + iter * self.rTz.strides[0])

    def step4_update_p(self, iter: int):
        wp.launch(
            update_cg_direction_kernel,
            dim=self.dim,
            inputs=[iter, self.z, self.rTz, self.p],
            outputs=[self.p],
            device=self.device,
        )

    # ------------------------------------------------------------ ITER7
    def step4_update_p_gated(self, iter: int):
        wp.launch(
            update_cg_direction_gated_kernel,
            dim=self.dim,
            inputs=[iter, self.z, self.rTz, self.p, self._stop_flag],
            outputs=[self.p],
            device=self.device,
        )

    def step7_update_x_r_gated(self, x: wp.array[wp.vec3], iter: int):
        wp.launch(
            step_cg_gated_kernel,
            dim=self.dim,
            inputs=[iter, self.rTz, self.pTAp, self.p, self.Ap, self._stop_flag],
            outputs=[x, self.r],
            device=self.device,
        )

    def step5_update_Ap(
        self,
        A_non_diag: SparseMatrixELL,
        A_diag: wp.array[Any],
        additional_Ap: wp.array[wp.vec3] = None,
    ):
        if additional_Ap is None:
            wp.launch(
                ell_mat_vec_mul_kernel,
                dim=self.dim,
                inputs=[A_non_diag, A_diag, self.p],
                outputs=[self.Ap],
                device=self.device,
            )
        else:
            wp.launch(
                ell_mat_vec_mul_add_kernel,
                dim=self.dim,
                inputs=[A_non_diag, A_diag, self.p, additional_Ap],
                outputs=[self.Ap],
                device=self.device,
            )

    def step6_update_pTAp(self, iter: int):
        array_inner(self.p, self.Ap, self.pTAp.ptr + iter * self.pTAp.strides[0])

    def step7_update_x_r(self, x: wp.array[wp.vec3], iter: int):
        wp.launch(
            step_cg_kernel,
            dim=self.dim,
            inputs=[iter, self.rTz, self.pTAp, self.p, self.Ap],
            outputs=[x, self.r],
            device=self.device,
        )

    def solve(
        self,
        A_non_diag: SparseMatrixELL,
        A_diag: wp.array[Any],
        x0: wp.array[wp.vec3],  # Pass `None` means x0[:] == 0.0
        b: wp.array[wp.vec3],
        inv_M: wp.array[Any],
        x1: wp.array[wp.vec3],
        iterations: int,
        additional_multiplier: Callable | None = None,
        preconditioner: Callable | None = None,
        eta: float = 0.0,
        fused_tp=None,
    ):
        # Prevent out-of-bounds in rTz/pTAp when iterations > maxIter.
        iterations = wp.min(iterations, self.maxIter)
        # ITER7: eta <= 0 -> the historical fixed-trip-count loop, bit-identical.
        adaptive = float(eta) > 0.0
        eta2 = float(eta) * float(eta)

        if x0 is None:
            x1.zero_()
        else:
            x1.assign(x0)

        if additional_multiplier is None:
            self.step1_update_r(A_non_diag, A_diag, b, x0)
        else:
            additional_Ax = additional_multiplier(x0) if x0 is not None else None
            self.step1_update_r(A_non_diag, A_diag, b, x0, additional_Ax)

        # FUSE1: the fused loop needs the TP data as a struct, so a caller
        # that supplies a `preconditioner` callable but no `fused_tp`
        # (tp_reduce="bucket") falls back; `adaptive` (ITER7) also falls back
        # because its device-side gate lives on the un-fused kernels.
        if self.fused and not adaptive and (preconditioner is None or fused_tp is not None):
            self._fused_loop(A_non_diag, A_diag, inv_M, x1, iterations, additional_multiplier, fused_tp)
            return

        for iter in range(iterations):
            self.step2_update_z(inv_M)
            if preconditioner is not None:
                preconditioner(self.r, self.z)
            self.step3_update_rTz(iter)
            if adaptive:
                # ITER7: decide (on device) whether THIS iteration still updates.
                wp.launch(
                    pcg_check_stop_kernel,
                    dim=1,
                    inputs=[iter, eta2, self.rTz],
                    outputs=[self._stop_flag, self._steps_taken],
                    device=self.device,
                )
                self.step4_update_p_gated(iter)
            else:
                self.step4_update_p(iter)

            if additional_multiplier is None:
                self.step5_update_Ap(A_non_diag, A_diag)
            else:
                additional_Ap = additional_multiplier(self.p)
                self.step5_update_Ap(A_non_diag, A_diag, additional_Ap)

            self.step6_update_pTAp(iter)
            if adaptive:
                self.step7_update_x_r_gated(x1, iter)
            else:
                self.step7_update_x_r(x1, iter)

        if adaptive:
            wp.launch(
                pcg_record_steps_kernel,
                dim=1,
                inputs=[self._steps_taken],
                outputs=[self.pcg_steps_hist],
                device=self.device,
            )

    # ------------------------------------------------------------- FUSE1
    def _fused_loop(self, A_non_diag, A_diag, inv_M, x1, iterations, additional_multiplier, fused_tp):
        """3 launches per PCG step (4 with the translation preconditioner).

        Entry contract: `self.r` already holds the initial residual.  See the
        FUSE1 block above `FusedTranslationPrecond` for the dependency argument
        and the numerics caveat (the two reductions become atomics).
        """
        if fused_tp is None:
            if self._fused_tp_off is None:
                self._fused_tp_off = FusedTranslationPrecond()
                self._fused_tp_off.enabled = 0
                self._fused_tp_off.n_comp = 0
                self._fused_tp_off.dt = 0.0
            tp = self._fused_tp_off
        else:
            tp = fused_tp
        tp_on = int(tp.enabled) != 0

        # FUSE2: the bucket rows are ACCUMULATED into, not assigned -- 2 memsets
        # per solve against ~3 launches per step saved.
        self.rTz_b.zero_()
        self.pTAp_b.zero_()

        # Prologue: z (+ rTz[0], or the TP coarse sums finished by K4).
        wp.launch(
            fused_z_kernel,
            dim=self.dim,
            inputs=[0, inv_M, self.r, tp],
            outputs=[self.z, self.rTz_b],
            device=self.device,
        )
        if tp_on:
            wp.launch(
                fused_tp_apply_kernel,
                dim=self.dim,
                inputs=[0, tp, self.r],
                outputs=[self.z, self.rTz_b],
                device=self.device,
            )

        for iter in range(iterations):
            wp.launch(
                fused_update_p_kernel,
                dim=self.dim,
                inputs=[iter, self.z, self.rTz_b, self.p],
                outputs=[self.p],
                device=self.device,
            )

            additional_Ap = None if additional_multiplier is None else additional_multiplier(self.p)
            wp.launch(
                fused_mat_vec_pTAp_kernel,
                dim=self.dim,
                inputs=[
                    iter,
                    A_non_diag,
                    A_diag,
                    self.p,
                    # unread when use_additional == 0; bind a live array anyway
                    self.p if additional_Ap is None else additional_Ap,
                    0 if additional_Ap is None else 1,
                    tp,
                ],
                outputs=[self.Ap, self.pTAp_b],
                device=self.device,
            )

            last = 1 if iter + 1 >= iterations else 0
            wp.launch(
                fused_step_z_kernel,
                dim=self.dim,
                inputs=[iter, 0 if last else 1, self.rTz_b, self.pTAp_b,
                        self.p, self.Ap, inv_M, tp],
                outputs=[x1, self.r, self.z],
                device=self.device,
            )
            if tp_on and not last:
                wp.launch(
                    fused_tp_apply_kernel,
                    dim=self.dim,
                    inputs=[iter + 1, tp, self.r],
                    outputs=[self.z, self.rTz_b],
                    device=self.device,
                )


if __name__ == "__main__":
    wp.init()
    dim = 100000
    diag_term = 5.0

    A_non_diag = SparseMatrixELL()
    A_diag = wp.zeros(dim, dtype=wp.float32)
    A_non_diag.num_nz = wp.zeros(dim, dtype=wp.int32)
    A_non_diag.nz_ell = wp.zeros(shape=(2, dim), dtype=NonZeroEntry)
    b = wp.zeros(dim, dtype=wp.vec3)
    x0 = wp.zeros(dim, dtype=wp.vec3)
    x1 = wp.zeros(dim, dtype=wp.vec3)
    wp.launch(generate_test_data_kernel, dim=dim, inputs=[dim, diag_term], outputs=[A_non_diag, A_diag, b, x0])

    inv_M = wp.array([1.0 / diag_term] * dim, dtype=float)

    solver = PcgSolver(dim, device="cuda:0")
    solver.solve(A_non_diag, A_diag, x0, b, inv_M, x1, iterations=30)

    rTr = wp.zeros(1, dtype=float)
    array_inner(solver.r, solver.r, rTr.ptr)
    print(rTr.numpy()[0])
